import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib.cm as cm
import matplotlib.colors as colors
import argparse
import pickle
import copy
from matplotlib import colors
from utils import bbox_iou, read_xml_annotations
import motmetrics as mm

parser = argparse.ArgumentParser()
parser.add_argument('--labels_dir', type=str, default='rcnn_detections_before_tuning/', help='')
parser.add_argument('--video_dir', type=str, default='../datasets/AICity_data/train/S03/c010/vdo.avi',help='')
parser.add_argument('--annotations_dir', type=str, default='./annotations/', help='')
parser.add_argument('--visualize', type=bool, default=False, help='')

def overlap_tracking(sorted_detections, video_dir, threshold = 0.5):

    assigned_id = 0
    track_dict = dict()
    whole_detections = []

    for key, value in sorted_detections.items():  # value[0] = bboxes, value[1] = scores, value[2] = ids
        current_frame_detections = []
        if int(key) != 1:
            for n_det, bbox in enumerate(value[0]):
                scores_and_ids = [bbox_iou(bbox, prev_frame_bbox) for prev_frame_bbox in whole_detections[int(key)-2]]
                max_in_list = max(scores_and_ids, key=lambda item: item[0])
                max_score = max_in_list[0]
                track_id = max_in_list[1]
                match_prev_bbox = max_in_list[2][1]

                # Check if the new bbox is more and less the same size as the previous one
                current_size = (bbox[0]+bbox[3])*(bbox[1]+bbox[2])
                prev_size = (match_prev_bbox[0]+match_prev_bbox[3])*(match_prev_bbox[1]+match_prev_bbox[2])

                if max_score >= threshold and current_size >= 0.8*prev_size:
                    current_frame_detections.append([track_id, bbox])
                else:
                    current_frame_detections.append([assigned_id, bbox])
                    assigned_id += 1
            whole_detections.append(current_frame_detections)
        else:
            for n_det, bbox in enumerate(value[0]):
                current_frame_detections.append([assigned_id, bbox])
                assigned_id += 1
            whole_detections.append(current_frame_detections)

    return whole_detections, assigned_id


def show_tracked_detections(tracked_detections, max_id, video_dir):
    capture = cv2.VideoCapture(video_dir)
    frame_idx = 0

    color_array = ["blue","black","brown","red","yellow","green","orange","beige","turquoise","pink"]

    dict_colors = dict()

    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_idx += 1

        for ind_detection in tracked_detections[frame_idx-1]:
            if ind_detection[0] not in dict_colors.keys():
                random.shuffle(color_array)
                random_color = colors.to_rgb(color_array[0])
                random_color = tuple([255 * x for x in random_color])
                dict_colors[ind_detection[0]] = random_color

            frame = cv2.rectangle(frame, (ind_detection[1][1], ind_detection[1][0]),
                                  (ind_detection[1][3], ind_detection[1][2]), dict_colors[ind_detection[0]], 3)
            frame = cv2.putText(frame, str(ind_detection[0]), (ind_detection[1][1]-5, ind_detection[1][0]-5),
                                cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, str(frame_idx), (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            
        cv2.namedWindow('output',cv2.WINDOW_NORMAL)
        cv2.imshow('output', frame)
        cv2.waitKey()



if __name__ == "__main__":
    """
    Load the labels (rois, scores, ids) for each frame
    and order them temporally
    """

    args, _ = parser.parse_known_args()
    path_labels = args.labels_dir
    path_video = args.video_dir
    visualize = args.visualize

    classes_of_interest = ['car', 'truck']

    detections_dict = dict()

    gt_bboxes = read_xml_annotations(args.annotations_dir)

    for filename in os.listdir(path_labels):
        detections = pickle.load(open(path_labels + filename, "rb"))
        cp_det = copy.deepcopy(detections)
        frame_idx = filename.split(".")[0]
        if frame_idx in detections_dict.keys():
            for box, score, category in zip(detections[0], detections[1], detections[2]):
                if category not in classes_of_interest:
                    cp_det[0].remove(box)
                    cp_det[1].remove(score)
                    cp_det[2].remove(category)
            detections_dict[frame_idx].append(cp_det)
        else:
            for box, score, category in zip(detections[0], detections[1], detections[2]):
                if category not in classes_of_interest:
                    cp_det[0].remove(box)
                    cp_det[1].remove(score)
                    cp_det[2].remove(category)
            detections_dict[frame_idx] = cp_det

    sorted_by_value = sorted(detections_dict.items(), key=lambda kv: int(kv[0]))

    sorted_detections = dict()
    for element in sorted_by_value:
        sorted_detections[element[0]] = element[1]

    tracked_detections, max_id = overlap_tracking(sorted_detections, path_video, threshold=0.75)

    if visualize is True:
        show_tracked_detections(tracked_detections, max_id, path_video)