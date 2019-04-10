import os
import cv2
import numpy as np
import argparse
from track_overlap import overlap_tracking, show_tracked_detections
from kalman_tracking import run_track
from utils import compute_idf1

parser   = argparse.ArgumentParser()
parser.add_argument('--tracking_type', type=str, default='overlap', help='')
parser.add_argument('--visualize', action='store_true', help='')
path_sequences = '../../../aic19-track1-mtmc-train/'


def start_tracking(detections_path, gt_file_path, video_dir, tracking_type='kalman', visualize_bool=False):

    with open(gt_file_path) as f:
        gt_data = f.readlines()

    cap = cv2.VideoCapture(video_dir)
    n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read and prepare groundtruth data
    dict_gt_bboxes = dict()
    for line in gt_data:
        splits = line.split(",")
        frame_idx = splits[0]
        x = float(splits[2])
        y = float(splits[3])
        w = float(splits[4])
        h = float(splits[5])
        conf = float(splits[6])
        id = int(splits[1])
        bbox = ['car',x,y,w,h,conf,id]
        if frame_idx in dict_gt_bboxes.keys():
            dict_gt_bboxes[frame_idx].append(bbox)
        else:
            dict_gt_bboxes[frame_idx] = [bbox]

    for n_frame in list(range(1, n_total_frames+1)):
        if str(n_frame) not in dict_gt_bboxes.keys():
            dict_gt_bboxes[str(n_frame)] = [[]]

    gt_list_sorted_by_key = sorted(dict_gt_bboxes.items(), key=lambda kv: int(kv[0]))
    gt_sorted = dict()
    for element in gt_list_sorted_by_key:
        gt_sorted[element[0]] = element[1]

    #################CHECK GT
    # list_check = list(gt_sorted.values())
    # capture = cv2.VideoCapture(video_dir)
    # frame_idx = 0
    # while True:
    #     success, frame = capture.read()
    #     if not success:
    #         break
    #     frame_idx += 1
    #     for ind_detection in list_check[int(frame_idx)-1]:
    #         if len(ind_detection) != 0:
    #             x = int(ind_detection[1])
    #             y = int(ind_detection[2])
    #             w = int(ind_detection[3])
    #             h = int(ind_detection[4])
    #             #frame = cv2.circle(frame,(x,y),20,(0,255,0),4)
    #             frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
    #     cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    #     cv2.imshow("output", frame)
    #     cv2.waitKey(1)
    ###################################
    ###########################################

    print(detections_path)
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
        if frame_idx in detections_dict.keys():
            detections_dict[frame_idx].append(bbox)
        else:
            detections_dict[frame_idx] = [bbox]

    for n_frame in list(range(1, n_total_frames+1)):
        if str(n_frame) not in detections_dict.keys():
            detections_dict[str(n_frame)] = [[]]

    det_list_sorted_by_key = sorted(detections_dict.items(), key=lambda kv: int(kv[0]))
    det_sorted = dict()
    for element in det_list_sorted_by_key:
        det_sorted[element[0]] = element[1]

    #################CHECK DETECTION
    # check_array = list(range(1, n_total_frames+1))
    # temp3 = [str(x) for x in check_array if str(x) not in list(detections_dict.keys())]
    # print(temp3)
    # list_check = list(detections_dict.values())
    # capture = cv2.VideoCapture(video_dir)
    # frame_idx = 0
    # while True:
    #     success, frame = capture.read()
    #     if not success:
    #         break
    #     frame_idx += 1
    #     for ind_detection in detections_dict.get(str(frame_idx)):
    #         if len(ind_detection) != 0:
    #             x = int(ind_detection[0])
    #             y = int(ind_detection[1])
    #             w = int(ind_detection[2])
    #             h = int(ind_detection[3])
    #             #frame = cv2.circle(frame,(x,y),20,(0,255,0),4)
    #             frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
    #     cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    #     cv2.imshow("output", frame)
    #     cv2.waitKey()
    ###################################

    tracked_detections = []
    if tracking_type.lower() == 'kalman':
        tracked_detections = run_track(video_dir, det_sorted, visualize=visualize_bool)
    elif tracking_type.lower() == 'overlap':
        tracked_detections, max_id = overlap_tracking(det_sorted)
        if visualize_bool:
            show_tracked_detections(tracked_detections, video_dir)

    else:
        print("THERE IS NOT TRACKER NAMED %s" % tracking_type)
        exit(0)

    list_gt_bboxes = list(gt_sorted.values())
    summary = compute_idf1(list_gt_bboxes, tracked_detections)
    print(video_dir + "--------->"+str(summary))


if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()

    if not os.path.exists(path_sequences):
        print('There is not such folder %s' % path_sequences)
        exit(0)

    for root, dirs, files in os.walk(path_sequences, topdown=True):
        for name in dirs:
            current_dir = os.path.join(root,name)
            if os.path.isdir(current_dir) and ('s03' in current_dir.lower()):
                if 'gt' in current_dir.lower():
                    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
                    gt_file_path = None
                    detections_path = None
                    for folder in os.listdir(parent_dir):
                        if os.path.isdir(os.path.join(parent_dir, folder)) and folder.lower() == 'gt':
                            _, _, gt_file = os.walk(os.path.join(parent_dir, folder)).__next__()
                            gt_file_path = parent_dir + os.sep + folder + os.sep + gt_file[0]

                        if os.path.isdir(os.path.join(parent_dir, folder)) and folder.lower() == 'det':
                            _, _, detections = os.walk(os.path.join(parent_dir, folder)).__next__()
                            detections_path = parent_dir + os.sep + folder + os.sep + detections[2]

                    (dirpath, dirnames, filenames) = os.walk(parent_dir).__next__()
                    print(os.path.join(parent_dir, filenames[1]))

                    # Filenames = roi.jpg, vdo.avi, calibration.txt

                    start_tracking(detections_path, gt_file_path, os.path.join(parent_dir, filenames[1]),
                                   FLAGS.tracking_type, FLAGS.visualize)
                    # if "c014" in parent_dir:
                    #     start_tracking(detections_path, gt_file_path, os.path.join(parent_dir, filenames[1]),
                    #                 FLAGS.tracking_type, FLAGS.visualize)
