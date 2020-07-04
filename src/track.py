from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    plot_arguments = []

    for path, img, img0 in dataloader:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets, ghost_tracks = tracker.update(blob, img0, opt)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if opt.vis_ghost_FP or opt.vis_ghost_FN:
            plot_arguments.append((img0, online_tlwhs, online_ids, frame_id,
                                  1. / timer.average_time, ghost_tracks))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    if opt.vis_ghost_FP or opt.vis_ghost_FN:
        return frame_id, timer.average_time, timer.calls, plot_arguments
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join('..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        print(data_root, seq)
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        if opt.vis_ghost_FP or opt.vis_ghost_FN:
            nf, ta, tc, plot_arguments = eval_seq(opt, dataloader, data_type, result_filename,
                                  save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        else:
            nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                                  save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        acc = evaluator.eval_file(result_filename)
        accs.append(acc)

        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)

        if opt.vis_ghost_FP:
            ghost_FP_image_dir = osp.join(opt.gfp_dir, 'images', opt.exp_dataset)
            if not osp.exists(ghost_FP_image_dir):
                os.makedirs(ghost_FP_image_dir)

            try:
                for img0, online_tlwhs, online_ids, frame_id, fps, ghost_boxes in plot_arguments:

                    ghost_im = vis.plot_boxes(img0, frame_id=frame_id, fps=fps, boxes=ghost_boxes, type='tlbr', color=(255,191,0), line_thickness=-1) # blue
                    FP_im = vis.plot_FP(img0, online_tlwhs, online_ids, acc.mot_events.loc[frame_id],
                                                     frame_id=frame_id, fps=fps, color=(0,0,255)) # red

                    # blend ghost image and FP image
                    alpha_ghost = 0.35
                    alpha_FP = 0.35
                    alpha_original = 1 - alpha_ghost - alpha_FP

                    blend_im = cv2.addWeighted(ghost_im, alpha_ghost, FP_im, alpha_FP, 0)
                    blend_im = cv2.addWeighted(img0, alpha_original, blend_im, 1 - alpha_original, 0)

                    # plot all GTs on blend image
                    blend_im = vis.plot_all_GT(blend_im, evaluator, frame_id=frame_id, fps=fps, color=(0,255,0)) # green
                    # plot all hypotheses on blend images
                    blend_im= vis.plot_hypotheses(blend_im, online_tlwhs, online_ids, color=(255,0,0)) # blue

                    cv2.imwrite(osp.join(ghost_FP_image_dir, '{:05d}.jpg'.format(frame_id)), blend_im)
            except:
                pass

            ghost_FP_video_dir = osp.join(opt.gfp_dir, 'video', opt.exp_dataset)
            if not osp.exists(ghost_FP_video_dir):
                os.makedirs(ghost_FP_video_dir)
            ghost_FP_video_path = osp.join(ghost_FP_video_dir, '{}_ghost_FP.mp4'.format(seq))

            cmd_str = 'ffmpeg -y -f image2 -i {}/%05d.jpg -c:v copy {}'.format(ghost_FP_image_dir, ghost_FP_video_path)
            os.system(cmd_str)
            os.system("rm -R {}".format(ghost_FP_image_dir))


        if opt.vis_ghost_FN:
            ghost_FN_image_dir = osp.join(opt.gfn_dir, 'images', opt.exp_dataset)
            if not osp.exists(ghost_FN_image_dir):
                os.makedirs(ghost_FN_image_dir)

            # try:
            if True:
                for img0, online_tlwhs, online_ids, frame_id, fps, ghost_boxes in plot_arguments:

                    ghost_im = vis.plot_boxes(img0, frame_id=frame_id, fps=fps, boxes=ghost_boxes, type='tlbr', color=(255,191,0), line_thickness=-1) # blue
                    FN_im = vis.plot_FN(img0, online_tlwhs, online_ids, acc.mot_events.loc[frame_id], evaluator,
                                                     frame_id=frame_id, fps=fps, color=(0,0,255)) # red

                    # blend ghost image and FN image
                    alpha_ghost = 0.35
                    alpha_FN = 0.35
                    alpha_original = 1 - alpha_ghost - alpha_FN

                    blend_im = cv2.addWeighted(ghost_im, alpha_ghost, FN_im, alpha_FN, 0)
                    blend_im = cv2.addWeighted(img0, alpha_original, blend_im, 1 - alpha_original, 0)

                    # plot all GTs on blend image
                    blend_im = vis.plot_all_GT(blend_im, evaluator, frame_id=frame_id, fps=fps, color=(0,255,0)) # green
                    # plot all hypotheses on blend images
                    blend_im= vis.plot_hypotheses(blend_im, online_tlwhs, online_ids, color=(255,0,0)) # blue

                    cv2.imwrite(osp.join(ghost_FN_image_dir, '{:05d}.jpg'.format(frame_id)), blend_im)
            # except:
            #     pass

            ghost_FN_video_dir = osp.join(opt.gfn_dir, 'video', opt.exp_dataset)
            if not osp.exists(ghost_FN_video_dir):
                os.makedirs(ghost_FN_video_dir)
            ghost_FN_video_path = osp.join(ghost_FN_video_dir, '{}_ghost_FN.mp4'.format(seq))

            cmd_str = 'ffmpeg -y -f image2 -i {}/%05d.jpg -c:v copy {}'.format(ghost_FN_image_dir, ghost_FN_video_path)
            os.system(cmd_str)
            os.system("rm -R {}".format(ghost_FN_image_dir))

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()

    if opt.exp_dataset == 'val_mot15_unique':
        seqs_str = '''ADL-Rundle-6
                      ADL-Rundle-8
                      KITTI-13
                      KITTI-17
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      Venice-2
                    '''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.exp_dataset == 'paper_val_mot15':
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.exp_dataset == 'val_mot16':
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/images/train')
    if opt.exp_dataset == 'test_mot16':
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        data_root = os.path.join(opt.data_dir, 'MOT16/images/test')
    if opt.exp_dataset == 'test_mot15':
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.exp_dataset == 'test_mot17':
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.exp_dataset == 'val_mot17':
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.exp_dataset == 'val_mot20':
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.exp_dataset == 'test_mot20':
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOT15_val_all_dla34',
         show_image=False,
         save_images=False,
         save_videos=False)
