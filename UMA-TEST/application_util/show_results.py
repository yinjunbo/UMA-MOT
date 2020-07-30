import argparse
import os
import cv2
import numpy as np
import colorsys
from image_viewer import ImageViewer


class Track:
    def __init__(self, track_bbox, track_id):
        self.track_bbox = track_bbox
        self.track_id = track_id


def gather_sequence_info(sequence_dir, det_dir, result_dir):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    # detection_dir = os.path.join(sequence_dir, "det", "det.txt")
    detection_dir = os.path.join(det_dir, '%s.txt' % os.path.basename(sequence_dir))
    # print(detection_dir)
    if os.path.exists(detection_dir):
        detections = np.loadtxt(detection_dir, delimiter=',')
    if os.path.exists(result_dir):
        results = np.loadtxt(result_dir, delimiter=',')
        # results = results[results[:, 6] == 1.0, :]
        # index = (results[:,-1]).tolist().index(min(results[:,-1]))
        # print(results[index])
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    # feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "update_ms": update_ms,
        "results": results
    }
    return seq_info


def frame_callback(vis, frame_idx, results):  # 跟踪过程

    print("Processing frame %05d" % frame_idx)
    # if frame_idx>300:
    frame_indices = results[:, 0].astype(np.int)
    mask = frame_indices == frame_idx
    record_trks = []
    for row in results[mask]:
        # bbox, confidence = row[2:6], row[6]
        bbox, id = row[2:6], int(row[1])
        record_trks.append(Track(bbox, id))
    image = cv2.imread(
        seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
    vis.set_image(image.copy())
    vis.draw_trackers(record_trks)


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self, seq_info, update_ms):
        image_shape = seq_info["image_size"][::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape = 1024, int(aspect_ratio * 1024)
        if update_ms is None:
            update_ms = seq_info["update_ms"]
        self.viewer = ImageViewer(
            update_ms, image_shape, "Figure %s" % seq_info["sequence_name"])
        self.viewer.thickness = 2
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]
        self.results = seq_info["results"]
        # self.detections = detections
        # self.iou = iou

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        frame_callback(self, self.frame_idx, self.results)
        self.frame_idx += 1
        return True

    def set_image(self, image):
        self.viewer.image = image

    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int), label=str(track_id))

    # def draw_detections(self, detections):
    #     self.viewer.thickness = 2
    #     self.viewer.color = 0, 0, 255
    #     for i, detection in enumerate(detections):
    #         self.viewer.rectangle(*detection.tlwh)

    def draw_trackers(self, tracks):
        self.viewer.thickness = 5
        for track in tracks:
            self.viewer.color = create_unique_color_uchar(track.track_id)
            self.viewer.rectangle(
                *track.track_bbox.astype(np.int), label=str(track.track_id))


def parse_args():

    parser = argparse.ArgumentParser(description="Siamese Tracking")
    parser.add_argument(
        "--sequence_dir", help="Path to the MOTChallenge sequence directory.",
        default=None, required=True)
    parser.add_argument(
        "--result_file", help="Tracking output in MOTChallenge file format.",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections (optional).",
        default=None)
    parser.add_argument(
        "--update_ms", help="Time between consecutive frames in milliseconds. "
        "Defaults to the frame_rate specified in seqinfo.ini, if available.",
        default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seq_info = gather_sequence_info(args.sequence_dir, args.detection_file, args.result_file)
    visualizer = Visualization(seq_info, update_ms=args.update_ms)
    visualizer.run(frame_callback)
