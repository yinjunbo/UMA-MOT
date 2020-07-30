import os.path as osp
import cv2
import numpy as np
import numba
from sklearn.utils.linear_assignment_ import linear_assignment
from .track import Track, TrackState
import tensorflow as tf
from tracker.Siamese_inference import inference_wrapper
from tracker.Siamese_inference.Siamese_tracker import Siamese_Tracker
from tracker.Siamese_utils.misc_utils import load_cfgs, get_center, auto_select_gpu


class MOT_Tracker:
    def __init__(self, max_age, occlusion_thres, association_thres):
        self.max_age = max_age
        self.occlusion_thres = occlusion_thres
        self.association_thres = association_thres
        self.siamese = None
        self.sess = None
        self.tracks = []  # save all the targets
        self._next_id = 1
        self.frame_rate = None

    @staticmethod
    def initiate_siamese_tracker(checkpoint, context_amount):
        model_config, _, track_config = load_cfgs(checkpoint)
        g = tf.Graph()
        with g.as_default():
            model = inference_wrapper.InferenceWrapper(context_amount)
            restore_fn = model.build_graph_from_config(model_config, track_config, checkpoint)
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        sess = tf.Session(graph=g, config=sess_config)
        restore_fn(sess)  # load ckpt
        siamese = Siamese_Tracker(model, model_config=model_config, track_config=track_config)
        return sess, siamese

    @staticmethod
    def npair_distance(a, b, data_is_normalized=False):
        result = np.zeros((a.shape[0], len(b)))
        frames_count = [ele.shape[0] for ele in b]
        b = np.vstack(b)
        a = np.reshape(a, (a.shape[0], -1))
        b = np.reshape(b, (b.shape[0], -1))
        if not data_is_normalized:
            a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
            b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
        concat_targets_result = np.dot(a, b.T)

        for k in range(a.shape[0]):
            index = 0
            for i, num in enumerate(frames_count):
                result[k, i] = np.mean(concat_targets_result[k, index:index + num])
                # result[k, i] = np.max(concat_targets_result[k, index:index + num])
                index += num
        return result

    @staticmethod
    @numba.jit
    def iou(bb_test, bb_gt):
        """
        Computes IUO between two bboxes in the form [x1,y1,x2,y2]
        """
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                  + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return (o)

    def associate_detections_to_trackers(self, detections, trackers, score_threshold, cos_matrix=None):

        """
        For tracked targets, update bboxes.
        For lost targets, associate with dets.
        """

        if (len(trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int), np.zeros((0, 0))

        if cos_matrix is not None:  # for lost targets
            association_matrix = cos_matrix
        else:  # for tracked targets
            association_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
            for d, det in enumerate(detections):
                for t, trk in enumerate(trackers):
                    association_matrix[d, t] = self.iou(det, trk)

        matched_indices = linear_assignment(-association_matrix)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if (d not in matched_indices[:, 0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if (t not in matched_indices[:, 1]):
                unmatched_trackers.append(t)

        # filter out matched with low score
        matches = []
        for m in matched_indices:
            if (association_matrix[m[0], m[1]] < score_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers), association_matrix

    def initiate_track(self, detection, current_frame_path):
        current_target_state = self.siamese.init_tracks(self.sess, detection, current_frame_path)
        self.tracks.append(Track(
            current_target_state, detection, self._next_id, self.max_age))
        self._next_id += 1

    def update(self, frame_path, checkpoint, context_amount, detections, iou):

        # init
        frame_count = int(osp.basename(frame_path).split(".")[0])
        if frame_count == 1:
            self.sess, self.siamese = self.initiate_siamese_tracker(checkpoint, context_amount)
        dets = np.array([d.tlwh for d in detections])  # dets: [x1,y1,w,h]
        if len(dets) == 0:
            dets_tlrb = dets
        else:
            dets_tlrb = dets.copy()
            dets_tlrb[:, 2:] += dets_tlrb[:, :2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]

        ###########################   STEP 1: occlusion detection   #######################

        for t in self.tracks:
            if t.is_tracked():
                similarity = t.current_target_state.similarity
                if similarity <= self.occlusion_thres or t.average_overlap < 0.5:
                    t.state = TrackState.Lost
                    t.current_target_state.bbox = t.current_target_state.old_bbox
                    t.current_target_state.scale_idx = t.current_target_state.old_scale_idx
                    t.current_target_state.search_pos = t.current_target_state.old_search_pos

        ###############################   STEP 2: update tracked targets   ###############################

        tracked_trks = [t for t in self.tracks if t.is_tracked()]
        trks = np.zeros((len(tracked_trks), 5))  # update tracked
        ret = []
        for t, trk in enumerate(trks):  # TODO: batch prediction
            tracked_trks[t].predict(self.sess, self.siamese, frame_path)
            pos = tracked_trks[t].track_bbox
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
        trks_tlrb = trks.copy()
        trks_tlrb[:, 2:4] += trks_tlrb[:, 0:2]  # convert to [xl,yl,w,h] to [x1,y1,x2,y2]
        matched, unmatched_dets, unmatched_trks, iou_matrix = self.associate_detections_to_trackers(dets_tlrb, trks_tlrb, iou) # associate with detection

        for t, trk in enumerate(tracked_trks): # refine bbox of matched track
            if t not in unmatched_trks:
                d = int(matched[np.where(matched[:, 1] == t)[0], 0])
                matched_iou = iou_matrix[d, t]
                det_state = self.siamese.init_tracks(self.sess, dets[d, :4], frame_path)  # TODO: batch prediction
                trk.update(dets[d, :4], det_state.reid_templates, 'tracked', matched_iou, self.frame_rate)
            else: # update unmatched track
                trk.overlap_history.append(0)
                history_len = len(trk.overlap_history)
                if history_len > 2 * self.frame_rate:
                    trk.overlap_history.pop(0)
                trk.average_overlap = sum(trk.overlap_history) / min(2 * self.frame_rate, history_len)

        ###############################    STEP 3: update lost targets  ###############################

        lost_trks = [t for t in self.tracks if t.is_lost()]
        for t in lost_trks:
            t.time_since_update += 1
        trks1 = np.zeros((len(lost_trks), 4))
        for t, trk in enumerate(trks1):
            pos = lost_trks[t].track_bbox
            trk[:] = [pos[0], pos[1], pos[2], pos[3]]
        trks_tlrb1 = trks1.copy()
        trks_tlrb1[:, 2:] += trks_tlrb1[:, :2]  # convert to [xl,yt,w,h] to [x1,y1,xr,yb]
        dets_tlwh1 = np.array([dets[i, :4] for i in unmatched_dets])  # [xl,yt,w,h]

        if not len(dets_tlwh1) == 0:
            dets_tlrb1 = dets_tlwh1.copy()
            dets_tlrb1[:, 2:] += dets_tlrb1[:, :2]
        else:
            dets_tlrb1 = dets_tlwh1

        if not (len(lost_trks) == 0 or len(dets_tlwh1) == 0):
            lost_trks_templates = [np.vstack(trk.current_target_state.his_feature) for trk in lost_trks]
            dets_reid_embeding, dets_trk_embeding = [], []
            for i in unmatched_dets:
                det_state = self.siamese.init_tracks(self.sess, dets[i, :4], frame_path)
                dets_reid_embeding.append(det_state.reid_templates)
                dets_trk_embeding.append(det_state.init_templates)
            dets_reid_embeding = np.array(dets_reid_embeding)
            cos_matrix = self.npair_distance(dets_reid_embeding, lost_trks_templates)  # todo: add motion model to limit detections to be matched
        else:
            cos_matrix = np.zeros((len(dets_tlrb1), len(trks_tlrb1)), dtype=np.float32)

        lost_matched, lost_unmatched_dets, lost_unmatched_trks, _ = self.associate_detections_to_trackers(dets_tlrb1, trks_tlrb1, self.association_thres, cos_matrix)

        # update recover targets
        for t, trk in enumerate(lost_trks):
            if (t not in lost_unmatched_trks):
                d = int(lost_matched[np.where(lost_matched[:, 1] == t)[0], 0])
                trk.update(dets_tlwh1[d, :4], [dets_reid_embeding[d], dets_trk_embeding[d]], 'recover')

        # initiate targets if not match with any dets
        for i in lost_unmatched_dets:
            self.initiate_track(dets_tlwh1[i, :4], frame_path)   # TODO: batch prediction

        ###############################   STEP 4: post processing   ###############################

        # remove dead tracklet
        wh = cv2.imread(frame_path).shape
        i = len(self.tracks)
        for trk in reversed(self.tracks):
            i -= 1
            if (trk.is_lost() and trk.time_since_update > self.max_age) or not trk.is_insight(
                    wh):
                self.tracks.pop(i)

        # record results
        for trk in reversed(self.tracks):
            d = trk.track_bbox
            if trk.is_tracked():
                ret.append(np.concatenate((d, [trk.track_id])).reshape(1, -1))  # +1 as MOT benchmark requires positive

        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))





