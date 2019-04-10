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
import matplotlib.pyplot as plt
import argparse
import pickle
import copy
from matplotlib import colors
from utils import bbox_iou_flow, read_xml_annotations, compute_idf1
import motmetrics as mm

parser = argparse.ArgumentParser()
parser.add_argument('--labels_dir', type=str, default='../code/week3/rcnn_detections_before_tuning/', help='')
# parser.add_argument('--labels_dir', type=str, default='../week3/predictions_cars/', help='')
parser.add_argument('--video_dir', type=str, default='../vdo.avi',help='')
parser.add_argument('--annotations_dir', type=str, default='./annotations/', help='')
parser.add_argument('--visualize', type=bool, default=False, help='')


def get_optical_flow(img1, img2, detections, frame_idx):
    flow_list = []

    feature_params = dict(maxCorners=500,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    det1_flow = []
    if img1 is not None:
        mask = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)
        for det in detections[0]:
            mask[det[0]:det[2], det[1]:det[3]] = 255
        # Debug
        # plt.imshow(mask, 'gray')
        # plt.axis('off')
        # plt.show()
        # plt.close()
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # p0 = cv2.goodFeaturesToTrack(img1_gray, mask=mask, **feature_params)
        # p1, st, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, p0, None, **lk_params)
        img1_gray_scaled = cv2.resize(img1_gray, (0,0), fx = 0.25, fy = 0.25)
        img2_gray_scaled = cv2.resize(img2_gray, (0,0), fx = 0.25, fy = 0.25)
        flow = cv2.calcOpticalFlowFarneback(img1_gray_scaled,img2_gray_scaled, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.resize(flow, (1920,1080))
        # print(flow.shape)
        # flow = p1-p0
        # flow = np.append(p0, flow, axis=2)

            # Select good points
        # good_new = p1[st==1]
        # good_old = p0[st==1]
        # # draw the tracks
        # color = np.random.randint(0,255,(100,3))
        # for i,(new,old) in enumerate(zip(good_new,good_old)):
        #     a,b = new.ravel()
        #     c,d = old.ravel()
        #     img1 = cv2.line(img1, (a,b),(c,d), (0,255,0), 2)
        #     # img1 = cv2.circle(img1,(a,b),5,color[80%(i+1)].tolist(),-1)
        img2_graycolor = cv2.cvtColor(img2_gray, cv2.COLOR_GRAY2BGR)
        hsv = np.zeros_like(img1)
        hsv[...,1] = 255

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        # flow_image = cv2.bitwise_and(bgr,bgr,mask = mask)
        flow_image = bgr
        result = cv2.addWeighted(img2_graycolor,0.3,flow_image,1,0)

        for ind_detection in detections[0]:
            result = cv2.rectangle(result, (ind_detection[1], ind_detection[0]),
                                    (ind_detection[3], ind_detection[2]), (0,0,255), 3)
        cv2.imshow('frame2',result)
        result = cv2.resize(result, (0,0), fx = 0.25, fy = 0.25)

        # cv2.imwrite('opt_frames/'+str(frame_idx)+'.png', result)
        # # cv2.imshow('frame',img1)
        k = cv2.waitKey(30)


        # print(flow)
        # Debug
        # show_optical_flow_arrows(img1, flow)

        for i in range(len(detections[0])):
            det_flow = flow[detections[0][i][0]:det[2], detections[0][i][1]:detections[0][i][3], :]
            accum_flow = np.mean(det_flow[np.logical_or(det_flow[:, :, 0] != 0, det_flow[:, :, 1] != 0), :], axis=0)
            if np.isnan(accum_flow).any():
                accum_flow = (0, 0)
            flow_list.append(accum_flow)

    return flow_list

def overlap_tracking(sorted_detections, threshold=0.5):

    assigned_id = 0
    track_dict = dict()
    whole_detections = []

    cap = cv2.VideoCapture('../vdo.avi')
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_idx = 0
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, old_frame = cap.read()

    for key, value in sorted_detections.items():  # value[0] = bboxes, value[1] = scores, value[2] = labels
        current_frame_detections = []
        if int(key) != 1:

            ret, frame = cap.read()
            frame_idx += 1
            # if frame_idx > 740:
                # break
            flow = get_optical_flow(old_frame, frame, sorted_detections[str(frame_idx)], frame_idx)

            old_frame = frame

            # print(flow)
            
            for n_det, bbox in enumerate(value[0]):
                # Build bbox
                confidence = value[1][n_det]
                label = value[2][n_det]
                bbox = [label]+bbox
                bbox.append(confidence)
                # bbox.append(flow[n_det])

                scores_and_ids = [ bbox_iou_flow(bbox[1:], prev_frame_bbox[1:]) for prev_frame_bbox in whole_detections[int(key)-2] ]
                max_in_list = max(scores_and_ids, key=lambda item: item[0])
                max_score = max_in_list[0]
                match_prev_bbox = max_in_list[1][0:4]
                track_id = max_in_list[1][-1]

                # Check if the new bbox is more and less the same size as the previous one
                current_size = (bbox[1]+bbox[4])*(bbox[2]+bbox[3])
                prev_size = (match_prev_bbox[0]+match_prev_bbox[3])*(match_prev_bbox[1]+match_prev_bbox[2])

                if max_score >= threshold and current_size >= 0.8*prev_size:
                    bbox.append(track_id)
                    current_frame_detections.append(bbox)
                else:
                    bbox.append(assigned_id)
                    current_frame_detections.append(bbox)
                    assigned_id += 1
            whole_detections.append(current_frame_detections)
        else:
            for n_det, bbox in enumerate(value[0]):
                # Build bbox
                confidence = value[1][n_det]
                label = value[2][n_det]
                bbox = [label]+bbox
                bbox.append(confidence)
                bbox.append(assigned_id)

                current_frame_detections.append(bbox)
                assigned_id += 1
            whole_detections.append(current_frame_detections)

    return whole_detections, assigned_id


def show_tracked_detections(tracked_detections, video_dir):
    capture = cv2.VideoCapture(video_dir)
    frame_idx = 0

    color_array = ["blue","black","brown","red","yellow","green","orange","beige","turquoise","pink"]

    dict_colors = dict()

    # frame_idx = 540
    # capture.set(cv2.CAP_PROP_POS_FRAMES, 540)

    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_idx += 1
        # if frame_idx > 190:
        #     break

        for ind_detection in tracked_detections[frame_idx-1]:
            if ind_detection[-1] not in dict_colors.keys():
                random.shuffle(color_array)
                random_color = colors.to_rgb(color_array[0])
                random_color = tuple([255 * x for x in random_color])
                dict_colors[ind_detection[-1]] = random_color

            frame = cv2.rectangle(frame, (ind_detection[2], ind_detection[1]),
                                  (ind_detection[4], ind_detection[3]), dict_colors[ind_detection[-1]], 3)
            frame = cv2.putText(frame, str(ind_detection[-1]), (ind_detection[2]-5, ind_detection[1]-5),
                                cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, str(frame_idx), (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

        frame = cv2.resize(frame,(0,0), fx=0.25, fy=0.25)
        cv2.imwrite('overlap_flow/image%04i.jpg' % frame_idx, frame)
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.imshow('output', frame)
        cv2.waitKey(10)


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
    min_confidence = 0.65

    for filename in os.listdir(path_labels):
        detections = pickle.load(open(path_labels + filename, "rb"))
        cp_det = copy.deepcopy(detections)
        frame_idx = filename.split(".")[0]
        if frame_idx in detections_dict.keys():
            for box, score, category in zip(detections[0], detections[1], detections[2]):
                if category not in classes_of_interest or score < min_confidence:
                    cp_det[0].remove(box)
                    cp_det[1].remove(score)
                    cp_det[2].remove(category)
            detections_dict[frame_idx].append(cp_det)
        else:
            for box, score, category in zip(detections[0], detections[1], detections[2]):
                if category not in classes_of_interest or score < min_confidence:
                    cp_det[0].remove(box)
                    cp_det[1].remove(score)
                    cp_det[2].remove(category)
            detections_dict[frame_idx] = cp_det

    sorted_by_value = sorted(detections_dict.items(), key=lambda kv: int(kv[0]))

    sorted_detections = dict()
    for element in sorted_by_value:
        sorted_detections[element[0]] = element[1]

    tracked_detections, max_id = overlap_tracking(sorted_detections, threshold=0.5)

    #### IDF1 computation
    list_gt_bboxes = list(gt_bboxes.values())
    summary = compute_idf1(list_gt_bboxes, tracked_detections)
    print(summary)

    if visualize is True:
        show_tracked_detections(tracked_detections, path_video)