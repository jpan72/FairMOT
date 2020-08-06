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

from .basetrack import BaseTrack, TrackState
import copy
from torchvision.transforms import transforms as T

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
        self.tlwh_buffer = deque([], maxlen=self.history_len)
        self.tlwh_buffer.append(new_track.tlwh)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

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
    def xyah_to_tlwh(xyah):
        ret = np.asarray(xyah).copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

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

        if opt.network == 'alexnet':
            input_size = 256
        else:
            input_size = 224

        self.transforms = T.Compose([
                T.ToPILImage(),
                T.Resize((input_size, input_size)),
                T.RandomCrop(200),
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

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

    def update(self, im_blob, img0, opt, gpn):
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
            # detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
            #               (tlbrs, f) in zip(dets[:, :5], id_feature)]
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
        #     if track.tracklet_len >= 30 and (track_h / track_w > 5 or (tlbr[0] < 0 or tlbr[1] < 0 or tlbr[2] > width or tlbr[3] > height)):
        #     # if track.tracklet_len > 5 and track_h / track_w > 4:
        #         # print('remove')
        #         # track.mark_removed()
        #         # self.removed_stracks.append(track)
        #
        #         track.update_ghost(track.tlwh, self.frame_id, update_feature=False)
        #         activated_stracks.append(track)
        #     else:
        #         new_strack_pool.append(track)

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
            #
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
                r_tracked_stracks = [r_tracked_stracks[it] for it in u_track if (it not in out_of_frame_it)] # 78.7, 76.1
                # r_tracked_stracks = [r_tracked_stracks[it] for it in u_track]
                dists = matching.iou_distance(r_tracked_stracks, detections_g)
                um_det_matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

                map1 = {}
                for um, det in um_det_matches:
                    map1[r_tracked_stracks[um]] = detections_g[det]

                # Run GPN
                for track, det in map1.items():

                    track_img = track.img_patch
                    det_img = det.img_patch
                    track_tlbr = track.tlbr
                    det_tlbr = det.tlbr
                    tlwh_history = track.tlwh_buffer

                    track_img = self.transforms(track_img)
                    det_img = self.transforms(det_img)

                    track_tlbr[0] /= 1088
                    track_tlbr[1] /= 608
                    track_tlbr[2] /= 1088
                    track_tlbr[3] /= 608

                    det_tlbr[0] /= 1088
                    det_tlbr[1] /= 608
                    det_tlbr[2] /= 1088
                    det_tlbr[3] /= 608

                    tlwh_history = np.array(list(tlwh_history))
                    tlwh_history[:, 0] /= 1088
                    tlwh_history[:, 1] /= 608
                    tlwh_history[:, 2] /= 1088
                    tlwh_history[:, 3] /= 608

                    track_img = track_img.cuda().float()
                    det_img = det_img.cuda().float()

                    track_tlbr = torch.tensor(track_tlbr).cuda().float()
                    det_tlbr = torch.tensor(det_tlbr).cuda().float()
                    tlwh_history = torch.tensor(tlwh_history).cuda().float()

                    delta_bbox_xyah = gpn(track_img.unsqueeze(0), det_img.unsqueeze(0),
                                     track_tlbr.unsqueeze(0), det_tlbr.unsqueeze(0),
                                     tlwh_history.unsqueeze(0))

                    delta_bbox_xyah = delta_bbox_xyah[0].cpu().detach().numpy()
                    delta_bbox_xyah[0] *= 1088
                    delta_bbox_xyah[1] *= 608
                    delta_bbox_xyah[3] *= 608

                    ghost_xyah = STrack.tlwh_to_xyah(track.tlwh) + delta_bbox_xyah
                    ghost_tlwh = STrack.xyah_to_tlwh(ghost_xyah)

                    track.update_ghost(ghost_tlwh, self.frame_id, update_feature=False)
                    track.update_ghost(track.tlwh, self.frame_id, update_feature=False)
                    track.ghost = True
                    activated_stracks.append(track)

                #
                # # Activate ghost tracks that get paired with ghost detections
                # for track, det in map1.items():
                #     track.update_ghost(track.tlwh, self.frame_id, update_feature=False)
                #     activated_stracks.append(track)
                #     # ghost_tlbrs.append(track.tlbr)

        # =========================================

        '''Mark unmatched tracks as lost'''
        for it in u_track:
            track = r_tracked_stracks[it]
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

        # """ Extra step: Remove out-of-frame tracks"""
        # for track in self.tracked_stracks:
        #     tlbr = track.tlbr
        #     tlwh = track.tlwh
        #     track_w, track_h = tlwh[2], tlwh[3]
        #     # import pdb; pdb.set_trace()
        #     # print(track.tracklet_len)
        #     # print(tlbr[2], width)
        #     # print(tlbr[3], height)
        #     # print()
        #     # print(tlbr)
        #
        #     # if track.tracklet_len > 10 and ( tlbr[0] < 0 or tlbr[1] < 0 or tlbr[2] > width or tlbr[3] > height): # 78.5, 76.1
        #     # if  (track.tracklet_len > 5 and track_h / track_w > 3): # 73.6, 75.9
        #     # if track.tracklet_len > 5 and (tlbr[0] < 0 or tlbr[1] < 0 or tlbr[2] > width or tlbr[3] > height): # 78.5, 76.1
        #     # if  (track.tracklet_len > 10 and track_h / track_w > 5): # 78.7, 76.1
        #     # print(track_h, track_w, track_h/track_w)
        #     # if track_h / track_w > 1:
        #     # print('...')
        #     if track.tracklet_len >= 5 and (track_h / track_w > 3 or (tlbr[0] < 0 or tlbr[1] < 0 or tlbr[2] > width or tlbr[3] > height)):
        #     # if track.tracklet_len > 5 and track_h / track_w > 4:
        #         # print('remove')
        #         # track.mark_removed()
        #         # self.removed_stracks.append(track)
        #
        #         track.update_ghost(track.tlwh, self.frame_id, update_feature=False)


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

        # in_frame_output_stracks = []
        # for track in output_stracks:
        #     tlbr = track.tlbr
        #     if tlbr[0] >= 0 and tlbr[1] >= 0 and tlbr[2] < width and tlbr[3] < height:
        #         in_frame_output_stracks.append(track)
        #
        # return in_frame_output_stracks, ghost_tlbrs

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
