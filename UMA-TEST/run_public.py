import os
import cv2
import numpy as np
import time
from application_util import visualization
from tracker.detection import Detection
from tracker.mot_tracker import MOT_Tracker
import configparser

def gather_sequence_info(sequence_dir, det_dir):
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
    detection_dir = os.path.join(det_dir, '%s.txt' % os.path.basename(sequence_dir))
    if os.path.exists(detection_dir):
        detections = np.loadtxt(detection_dir, delimiter=',')
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

    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx):
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence = row[1:5], row[5]
        detection_list.append(Detection(bbox, confidence))
    return detection_list


def run(sequence_dir, det_dir, checkpoint, output_file,
             max_age, context_amount, iou,
            occlusion_thres, association_thres, display):

    seq_info = gather_sequence_info(sequence_dir, det_dir)
    tracker = MOT_Tracker(max_age, occlusion_thres, association_thres)

    conf = configparser.ConfigParser()
    conf.read(os.path.join(sequence_dir, 'seqinfo.ini'))
    tracker.frame_rate = int(conf.get('Sequence', 'frameRate'))

    if display:
        gt = seq_info['groundtruth'][np.where(seq_info['groundtruth'][:,-2] == 1)[0], :]

    results = []
    runtime = []
    detections = seq_info["detections"]
    def frame_callback(vis, detections, frame_idx, iou):

        # Load image and generate detections.
        print("Processing frame %05d" % frame_idx)
        frame_path = seq_info['image_filenames'][int(frame_idx)]
        detections = create_detections(detections, frame_idx)

        # Update tracker.
        before_time = time.time()
        trackers = tracker.update(frame_path, checkpoint, context_amount, detections, iou)  # tracking
        runtime.append(time.time()-before_time)
        # Store results.
        for d in trackers:
            results.append([
                frame_idx, d[4], d[0], d[1], d[2], d[3]])

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_groundtruth(gt[np.where(gt[:,0] == frame_idx)[0], 1], \
                                gt[np.where(gt[:,0] == frame_idx)[0], 2: 6])
            record_trks = [t for t in tracker.tracks if (t.is_tracked() and t.time_since_update <= 5)]
            vis.draw_trackers(record_trks)

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, detections, iou, update_ms=100)
    else:
        visualizer = visualization.NoVisualization(seq_info, detections, iou,)

    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]), file=f)

    return sum(runtime) / len(runtime) * 1000

