from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from models import *
from models.decode import mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat
from tracker import matching
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *
from utils.post_process import ctdet_post_process
from tracking_utils.io import unzip_objs
import os.path as osp

from .basetrack import BaseTrack, TrackState
import copy

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30, img_patch=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.is_ghost = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

        self.history_len = 10
        self.tlwh_buffer = deque([], maxlen=self.history_len)
        self.img_patch = img_patch

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.tlwh_buffer = deque([], maxlen=self.history_len)
        self.tlwh_buffer.append(new_track.tlwh)

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        # self.is_ghost = False # .......... turned off for now

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)
        self.tlwh_buffer.append(new_track.tlwh)
        self.img_patch = new_track.img_patch

    def update_ghost(self, ghost_tlwh, frame_id, update_feature=True, var_multiplier=1):
        """
        Update a matched track with GPN regressed coords
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = ghost_tlwh
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh), var_multiplier)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.is_ghost = True

        # self.score = new_track.score
        # if update_feature:
        #     self.update_features(new_track.curr_feat)
        self.tlwh_buffer.append(ghost_tlwh)

    def update_FN(self, FN_tlwh, frame_id, FN_img_patch, update_feature=False, update_kf=True, var_multiplier=1):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        if update_kf:
            new_tlwh = FN_tlwh
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.tlwh_buffer.append(FN_tlwh)

        self.score = 1
        self.img_patch = FN_img_patch #?

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets_prev = dets
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        # import pdb; pdb.set_trace()

        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def update(self, im_blob, img0, opt, evaluator, path):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]  # 1920
        height = img0.shape[0]  # 1080
        inp_height = im_blob.shape[2]  # 608
        inp_width = im_blob.shape[3]  # 1088
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        # import pdb; pdb.set_trace()
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            # import pdb; pdb.set_trace()
            dets, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)  # dets: in 272x152 scale
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        # Get image patch
        bbox = np.rint(dets[:, :4]).astype(int)
        bbox_x, bbox_y, bbox_w, bbox_h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        h0, w0, _ = img0.shape

        bbox_x[bbox_x > w0] = w0
        bbox_x[bbox_x < 0] = 0
        bbox_y[bbox_y > h0] = h0
        bbox_y[bbox_y < 0] = 0

        h_rs = 224
        w_rs = 224
        img_patches = []
        for x, y, w, h in zip(bbox_x, bbox_y, bbox_w, bbox_h):
            tmp = img0[y:y + h, x:x + w, :]
            tmp = np.ascontiguousarray(tmp)
            import cv2
            tmp = cv2.resize(tmp, (h_rs, w_rs))
            img_patches.append(tmp)

        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30, p) for
                          (tlbrs, f, p) in zip(dets[:, :5], id_feature, img_patches)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)

        # ''' Extend tracks moving out of frame with KF prediction '''
        # new_strack_pool = []
        # for track in strack_pool:
        #     tlbr = track.tlbr
        #     tlwh = track.tlwh
        #     track_w, track_h = tlwh[2], tlwh[3]
        #
        #     if track.tracklet_len >= 20 and (track_h / track_w > 10 or (tlbr[0] < 0 or tlbr[1] < 0 or tlbr[2] > width or tlbr[3] > height)):
        #     # if track.tracklet_len > 5 and track_h / track_w > 4:
        #         # print('remove')
        #         # track.mark_removed()
        #         # self.removed_stracks.append(track)
        #
        #         track.update_ghost(track.tlwh, self.frame_id, update_feature=False)
        #         activated_stracks.append(track)
        #     else:
        #         new_strack_pool.append(track)
        #
        # strack_pool = new_strack_pool

        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        ghost_dets = []
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)

                # *** For each feature matched detection, create a ghost detection ***
                ghost_det = copy.deepcopy(det)
                ghost_dets.append(ghost_det)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        detections = [detections[i] for i in u_detection]

        # =========== ghost matching ===========
        if opt.ghost:

            # remove tracks that are out of frame (i.e. don't match ghosts with such tracks)
            out_of_frame_it = []
            for it in u_track:
                track = r_tracked_stracks[it]
                tlbr = track.tlbr
                if track.tracklet_len < 1 or (tlbr[0] < 0 or tlbr[1] < 0 or tlbr[2] > width or tlbr[3] > height):
                    track.mark_lost()
                    lost_stracks.append(track)
                    out_of_frame_it.append(it)
                    # track.mark_removed()
                    # removed_stracks.append(track)

            detections_g = ghost_dets

            # r_tracked_stracks = [r_tracked_stracks[it] for it in u_track if (it not in out_of_frame_it)]


            '''Occlusion Reasoning to Find Ghost BBoxes'''
            # ghost_tlbrs = []

            if self.frame_id > 1:

                # Match unmatched tracks with matched dets (foreground tracklets)
                # r_tracked_stracks = [r_tracked_stracks[it] for it in u_track if (it not in out_of_frame_it) and r_tracked_stracks[it].tracklet_len > 5]
                r_tracked_stracks = [r_tracked_stracks[it] for it in u_track if (it not in out_of_frame_it)] # 78.7, 76.1
                # r_tracked_stracks = [r_tracked_stracks[it] for it in u_track]
                if False:
                    dists = matching.iou_distance(r_tracked_stracks, detections_g)
                    # um_det_matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
                    um_det_matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

                    map1 = {}
                    for um, det in um_det_matches:
                        map1[r_tracked_stracks[um]] = detections_g[det]
                    # if len(map1) > 0:
                    #     print(map1.keys())

                    # Activate ghost tracks that get paired with ghost detections
                    for track, det in map1.items():
                        track.update_ghost(track.tlwh, self.frame_id, update_feature=False)
                        activated_stracks.append(track)
                        # ghost_tlbrs.append(track.tlbr)

        # =========================================
        ''' Collect data for GPN'''

        prefix = path.split('img1')[0]
        # if opt.use_featmap:
        if False:
            dataset_root = '../preprocess-ghost-bbox-fairmot-conf0.8-occ{}/'.format(opt.occ_reason_thres)
        else:
            dataset_root = '../preprocess-ghost-bbox-th0.5-fairmot-conf{}/'.format(opt.conf_thres)
        save_dir = osp.join(prefix, 'preprocess').replace(opt.data_dir, dataset_root)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        save_path = path.replace(opt.data_dir, dataset_root).replace('img1', 'preprocess').replace('.png', '').replace('.jpg', '')

        cur_output_stracks = [track for track in activated_stracks if
                          track.is_activated and track.state == TrackState.Tracked]
        trk_tlwhs = [track.tlwh for track in cur_output_stracks]
        trk_ids = [track.track_id for track in cur_output_stracks]
        # trk_ids = np.arange(len(trk_tlwhs))
        print(self.frame_id)
        evaluator.eval_frame(self.frame_id, trk_tlwhs, trk_ids, rtn_events=False)

        FN_tlbrs_selected = []
        tracks_selected = []

        before_boxes = []
        after_boxs = []

        # frame_id:       1, 2, 3, 4
        # acc_event key:     0, 1, 2 #把上面3的结果提前拿到acc=1里去测试哪些missing
        # gt dict(1-idx):    1, 2, 3

        if self.frame_id > 2: # = 3

            acc_frame = evaluator.acc.mot_events.loc[self.frame_id-2] # =1 ok
            miss_rows = acc_frame[acc_frame.Type.eq('MISS')]
            miss_OIds = miss_rows.OId.values

            gt_objs = evaluator.gt_frame_dict.get(self.frame_id-1, []) # =2
            gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

            acc_frame_p = evaluator.acc.mot_events.loc[self.frame_id-3] # =0 ok

            # print('miss_OIds', miss_OIds)
            # Go over missing tracks by their OId
            found_rtracks = []
            for miss_OId in miss_OIds:

                FN_tlwh = gt_tlwhs[gt_ids==miss_OId][0]
                FN_tlbrs_selected.append(STrack.tlwh_to_tlbr(FN_tlwh))

                miss_HId_p = acc_frame_p[acc_frame_p.OId.eq(miss_OId)].HId.values
                if len(miss_HId_p) == 0:
                    print('cannot find miss OId from tracks in previous frame')
                    continue
                else:
                    miss_HId_p = miss_HId_p[0]
                # import pdb; pdb.set_trace()
                track_id = miss_HId_p

                track = None
                for x in r_tracked_stracks:
                    if x.track_id == track_id:
                        track = x
                        found_rtracks.append(track_id)

                if track is None:
                    continue
                #
                # if track == None or track.track_id not in map1:
                #     # print('cannot find track ID {} from lost tracks in current frame (did not get matched during ghost match)')
                #     continue
                #
                # det = map1[track_id]

                det = [0,0,0,0] # dummy

                target_delta_bbox = FN_tlwh - track.tlwh

                before_boxes.append(track.tlwh)
                after_boxs.append(FN_tlwh)

                print('!!!!!!')
                # np.savez(save_path, track_feat=track.img_patch, det_feat=det.img_patch,
                #          track_tlbr=track.tlbr, det_tlbr=det.tlbr, tlwh_history=track.tlwh_buffer,
                #          target_delta_bbox=target_delta_bbox)
                np.savez(save_path, track_feat=track.img_patch, det_feat=track.img_patch,
                         track_tlbr=track.tlbr, det_tlbr=track.tlbr, tlwh_history=track.tlwh_buffer,
                         target_delta_bbox=target_delta_bbox)

                # print('except, updating with FN tlwh')
                # add FN to the tracked pool
                x,y,w,h = FN_tlwh.astype(int)
                track.update_FN(FN_tlwh, self.frame_id, img0[y:y+h, x:x+w, :]) ###???
                activated_stracks.append(track)

        # # ================================================
        # '''Mark unmatched tracks as lost'''
        # for it in u_track:
        #     if not track.state == TrackState.Lost:
        #         track.mark_lost()
        #         lost_stracks.append(track)

        for track in r_tracked_stracks:
            if track.track_id not in found_rtracks:
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)


        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)




        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated and track.state == TrackState.Tracked]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_stracks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        ghost_tlbrs = [track.tlbr for track in self.tracked_stracks if track.is_activated and track.is_ghost]


        # ''' Collect data for GPN'''
        #
        # prefix = path.split('img1')[0]
        # # if opt.use_featmap:
        # if False:
        #     dataset_root = '../preprocess-ghost-bbox-fairmot-conf0.8-occ{}/'.format(opt.occ_reason_thres)
        # else:
        #     dataset_root = '../preprocess-ghost-bbox-th0.5-fairmot-conf{}/'.format(opt.conf_thres)
        # save_dir = osp.join(prefix, 'preprocess').replace(opt.data_dir, dataset_root)
        # if not osp.exists(save_dir):
        #     os.makedirs(save_dir)
        # save_path = path.replace(opt.data_dir, dataset_root).replace('img1', 'preprocess').replace('.png', '').replace('.jpg', '')
        #
        #
        # trk_tlwhs = [track.tlwh for track in output_stracks]
        # trk_ids = [track.track_id for track in output_stracks]
        # # trk_ids = np.arange(len(trk_tlwhs))
        # evaluator.eval_frame(self.frame_id, trk_tlwhs, trk_ids, rtn_events=False)
        #
        # FN_tlbrs_selected = []
        # tracks_selected = []
        #
        # before_boxes = []
        # after_boxs = []
        #
        # if self.frame_id > 2:
        #
        #     acc_frame = evaluator.acc.mot_events.loc[self.frame_id-1]
        #     miss_rows = acc_frame[acc_frame.Type.eq('MISS')]
        #     miss_OIds = miss_rows.OId.values
        #
        #     gt_objs = evaluator.gt_frame_dict.get(self.frame_id, [])
        #     gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]
        #
        #     acc_frame_p = evaluator.acc.mot_events.loc[self.frame_id-2]
        #
        #     # print('miss_OIds', miss_OIds)
        #     # Go over missing tracks by their OId
        #     for miss_OId in miss_OIds:
        #
        #         FN_tlwh = gt_tlwhs[gt_ids==miss_OId][0]
        #         FN_tlbrs_selected.append(STrack.tlwh_to_tlbr(FN_tlwh))
        #
        #         miss_HId_p = acc_frame_p[acc_frame_p.OId.eq(miss_OId)].HId.values
        #         if len(miss_HId_p) == 0:
        #             print('cannot find miss OId from tracks in previous frame')
        #             continue
        #         else:
        #             miss_HId_p = miss_HId_p[0]
        #
        #         track_id = miss_HId_p
        #
        #         track = None
        #         for x in r_tracked_stracks:
        #             if x.track_id == track_id:
        #                 track = x
        #
        #         if track is None:
        #             continue
        #         #
        #         # if track == None or track.track_id not in map1:
        #         #     # print('cannot find track ID {} from lost tracks in current frame (did not get matched during ghost match)')
        #         #     continue
        #         #
        #         # det = map1[track_id]
        #
        #         det = [0,0,0,0] # dummy
        #
        #         # print(self.frame_id)
        #         # print(FN_tlwh, track.mean[:4].astype(np.int), det.tlbr)
        #         target_delta_bbox = FN_tlwh - track.tlwh
        #
        #         before_boxes.append(track.tlwh)
        #         after_boxs.append(FN_tlwh)
        #
        #         print('!!!!!!')
        #         # np.savez(save_path, track_feat=track.img_patch, det_feat=det.img_patch,
        #         #          track_tlbr=track.tlbr, det_tlbr=det.tlbr, tlwh_history=track.tlwh_buffer,
        #         #          target_delta_bbox=target_delta_bbox)
        #         np.savez(save_path, track_feat=track.img_patch, det_feat=track.img_patch,
        #                  track_tlbr=track.tlbr, det_tlbr=track.tlbr, tlwh_history=track.tlwh_buffer,
        #                  target_delta_bbox=target_delta_bbox)
        #
        #         # print('except, updating with FN tlwh')
        #         # add FN to the tracked pool
        #         x,y,w,h = FN_tlwh.astype(int)
        #         track.update_FN(FN_tlwh, self.frame_id, img0[y:y+h, x:x+w, :])
        #         activated_stracks.append(track)


        return output_stracks, ghost_tlbrs

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
