import os
import sys
import cv2
import numpy as np
import argparse
from track_overlap import overlap_tracking, show_tracked_detections
from kalman_tracking import run_track
sys.path.insert(0, '../MTSC')
from utils import compute_idf1

parser   = argparse.ArgumentParser()
parser.add_argument('--tracking_type', type=str, default='kalman', help='')
parser.add_argument('--visualize', action='store_false', help='')
path_sequences = '../../aic19-track1-mtmc-train/'


def get_detections(detections_path, gt_file_path, min_area=None):

    with open(gt_file_path) as f:
        gt_data = f.readlines()

    # Read and prepare groundtruth data
    dict_gt_bboxes = dict()
    for line in gt_data:
        splits = line.split(",")
        frame_idx = splits[0]
        id = int(splits[1])
        x = float(splits[2])
        y = float(splits[3])
        w = float(splits[4])
        h = float(splits[5])
        conf = float(splits[6])
        bbox = [x,y,w,h,conf,id]
        if frame_idx in dict_gt_bboxes.keys():
            dict_gt_bboxes[frame_idx].append(bbox)
        else:
            dict_gt_bboxes[frame_idx] = [bbox]

    ###########################################

    with open(detections_path) as f:
        detections_data = f.readlines()

    detections_dict = dict()
    for line in detections_data:
        splits = line.split(",")
        frame_idx = splits[0]
        x = float(splits[2])
        y = float(splits[3])
        w = float(splits[4])
        h = float(splits[5])
        conf = float(splits[6])
        bbox = [x,y,w,h,conf]
        if (w*h) > 7000 and conf > 0.6:
            if frame_idx in detections_dict.keys():
                detections_dict[frame_idx].append(bbox)
            else:
                detections_dict[frame_idx] = [bbox]

    return dict_gt_bboxes, detections_dict


if __name__ == "__main__":
    for sequence in ['train/S01', 'train/S04']:
        camera_dirs = os.listdir(path_sequences + sequence)
        # tracked_detections = []
        for camera_dir in camera_dirs:
            dir = path_sequences + sequence + '/' + camera_dir
            video_dir = dir + '/vdo.avi'
            gt_path = dir + '/gt/gt.txt'
            detections_path = dir + '/det/det_mask_rcnn.txt'

            gt_bboxes, detections_bboxes = get_detections(detections_path, gt_path)

            tracked = run_track(video_dir, detections_bboxes, False, wait_time=50)
            print(tracked)
            # tracked_detections, ids = overlap_tracking(gt_bboxes)
            # show_tracked_detections(tracked_detections, video_dir)



