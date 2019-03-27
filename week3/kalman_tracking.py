from __future__ import print_function
import os
import argparse
import pickle

import numpy as np
import cv2
from sort import Sort
from utils import read_xml_annotations, compute_idf1

parser   = argparse.ArgumentParser()

parser.add_argument('--video_dir', type=str, default='../datasets/AICity_data/train/S03/c010/vdo.avi', help='')
parser.add_argument('--labels_dir', type=str, default='rcnn_detections_before_tuning/', help='')
parser.add_argument('--visualize', type=bool, default=False, help='')
parser.add_argument('--annotations_dir', type=str, default='./annotations/', help='')

classes_of_interest = ['car', 'bus', 'truck']
min_confidence      = 0.65


def read_detections(labels_file):
    with open(labels_file, "rb") as fp:
        labels = pickle.load(fp)

    bounding_boxes, scores, classes = labels

    detections = []
    for box, score, category in zip(bounding_boxes, scores, classes):
        if score >= min_confidence and category in classes_of_interest:
            detections.append(box)

    return np.array(detections)


def main(video_dir, labels_dir, visualize):

    capture = cv2.VideoCapture(video_dir)

    frame_idx = 0
    kalman_tracker = Sort()

    whole_video_detections = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_idx += 1
        labels_file = os.path.join(labels_dir, str(frame_idx)+'.pickle')
        detections = read_detections(labels_file)
        trackers = kalman_tracker.update(detections)

        current_frame_detections = []
        for track_det in trackers:
            track_det = track_det.astype(np.uint32)
            current_frame_detections.append(['car', track_det[0], track_det[1], track_det[2],
                                             track_det[3], track_det[4]])

            if visualize is True:
                cv2.rectangle(frame, (track_det[1], track_det[0]), (track_det[3], track_det[2]), (0, 0, 255), 3)

                font = cv2.FONT_HERSHEY_DUPLEX
                placement = (track_det[3] + 10, track_det[2] + 10)
                font_scale = 1
                font_color = (0, 255, 0)
                line_type = 2

                cv2.putText(frame, str(track_det[4]), placement, font, font_scale, font_color, line_type)

        whole_video_detections.append(current_frame_detections)

        if visualize is True:
            cv2.imwrite('tracking_videos/kalman/image%04i.jpg' % frame_idx, frame)
            cv2.imshow('output', frame)
            cv2.waitKey(10)

    return whole_video_detections


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    visualize = FLAGS.visualize
    tracked_detections = main(FLAGS.video_dir, FLAGS.labels_dir, visualize)

    # Compute idf1
    gt_bboxes = read_xml_annotations(FLAGS.annotations_dir)
    list_gt_bboxes = list(gt_bboxes.values())
    summary = compute_idf1(list_gt_bboxes, tracked_detections)
    print(summary)


    print("exiting")


