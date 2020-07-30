from tracker.Siamese_utils.infer_utils import convert_bbox_format, Rectangle
import numpy as np


class TrackState:

    Tracked = 1
    Lost = 2

class Track:

    def __init__(self, current_target_state, track_bbox, track_id, max_age):
        self.current_target_state = current_target_state
        self.track_bbox = track_bbox
        self.track_id = track_id
        self.time_since_update = 0
        self.state = TrackState.Tracked
        self._max_age = max_age
        self.overlap_history = [1]
        self.average_overlap = 1

    def predict(self, sess, siamese, frame_path):

        self.current_target_state, self.track_bbox = siamese.track(sess, self.current_target_state, frame_path)
        self.time_since_update += 1

    def update(self, detection, det_embeding, mode, matched_iou=1.0, frame_rate=30):
        self.time_since_update = 0

        if mode == 'tracked':
            self.overlap_history.append(1 if matched_iou > 0.5 else 0)
            history_length = len(self.overlap_history)
            if history_length > 2 * frame_rate:
                self.overlap_history.pop(0)
            self.average_overlap = sum(self.overlap_history) / min(2 * frame_rate, history_length)

            refine_detection = [0.5 * self.track_bbox[0] + 0.5 * detection[0],
                                0.5 * self.track_bbox[1] + 0.5 * detection[1],
                                0.5 * self.track_bbox[2] + 0.5 * detection[2],
                                0.5 * self.track_bbox[3] + 0.5 * detection[3]]  # ltrb

            self.current_target_state.bbox = convert_bbox_format(Rectangle(refine_detection[0],
                                          refine_detection[1],
                                          refine_detection[2],
                                          refine_detection[3]), 'center-based')
            self.track_bbox = np.array(refine_detection)  # track result
            self.current_target_state.his_feature.append(det_embeding)
            if len(self.current_target_state.his_feature) > frame_rate:
                self.current_target_state.his_feature.pop(0)

            return

        if mode == 'recover':

            self.state = TrackState.Tracked   # re-tracked
            init_bb = Rectangle(int(detection[0]) - 1, int(detection[1]) - 1, int(detection[2]), int(detection[3]))  # xl, yt, w, h
            bbox = convert_bbox_format(init_bb, 'center-based')
            self.current_target_state.bbox = bbox
            self.current_target_state.reid_templates = det_embeding[0]
            self.current_target_state.init_templates = det_embeding[1]
            self.current_target_state.scale_idx = int(1)
            self.current_target_state.similarity = 1.0
            self.current_target_state.original_target_wh = [bbox.width, bbox.height]
            self.current_target_state.bbox_in = detection
            self.track_bbox = np.array([init_bb.x, init_bb.y, init_bb.width, init_bb.height])

            self.overlap_history = [1]
            self.average_overlap = 1
            self.current_target_state.his_feature = []
            self.current_target_state.his_feature.append(self.current_target_state.reid_templates)

    def is_tracked(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Tracked

    def is_lost(self):
        """Returns True if this track is lost."""
        return self.state == TrackState.Lost


    def is_insight(self, shape):
        x = self.track_bbox[0] + self.track_bbox[2] / 2 
        y = self.track_bbox[1] + self.track_bbox[3] / 2
        return x > 1 and y > 1 and x < shape[1] and y < shape[0]


