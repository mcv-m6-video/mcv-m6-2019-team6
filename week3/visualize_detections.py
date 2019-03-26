import numpy as np
import cv2
import pickle
from coco_class_names import class_names
import argparse
import os
import random
from utils import read_xml_annotations

parser   = argparse.ArgumentParser()

parser.add_argument('--video_dir', type=str, default='../datasets/AICity_data/train/S03/c010/vdo.avi',help='')
parser.add_argument('--labels_dir', type=str, default='rcnn_detections_before_tuning/',help='')


classes_of_interest = ['car','bus','truck']
min_confidence      = 0.8

def draw_boxes(img, bounding_boxes, scores, classes, min_confidence):
    for box, score,category in zip(bounding_boxes, scores, classes):
        if score >= min_confidence and category in classes_of_interest:
            cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(0,0,255),3)		
    return img


def draw_boxes_gt(img, bounding_boxes):
    for box in bounding_boxes:
        _, xtl, ytl, height, width, _, id_det = box
        cv2.rectangle(img,(xtl,ytl),(xtl+width,ytl+height),(255,0,0),3)		
    return img


def save_detections_for_map(image, im_idx, bounding_boxes, scores, gt_boxes):
    detection_results_dir = 'mAP/input/detection-results/'
    ground_truth_dir      = 'mAP/input/ground-truth/'
    images_dir            = 'mAP/input/images-optional/'

    # -------------- save image
    #cv2.imwrite(os.path.join(images_dir,str(im_idx)+'.jpeg'),image)
    # -------------- save gorund truth
    labels_file = os.path.join(ground_truth_dir, str(im_idx)+'.txt')
    file = open(labels_file, "w") 

    for box in gt_boxes:
        _, xtl, ytl, height, width, _, id_det = box
        label = [xtl,ytl,xtl+width,ytl+height]
        label = [str(e) for e in label]
        file.write('car'+' '+label[0]+' '+label[1]+' '+label[2]+' '+label[3]+'\n')
    file.close() 
    # -------------- save detection        
    dets_file = os.path.join(detection_results_dir, str(im_idx)+'.txt')
    file = open(dets_file, "w") 

    for b, s in zip(bounding_boxes, scores):
        b = [str(e) for e in b]
        file.write('car'+' '+str(s)+' '+b[1]+' '+b[0]+' '+b[3]+' '+b[2]+'\n')
    file.close()         


def demo(video_dir, labels_dir):
    detection_results_dir = 'mAP/input/detection-results/'
    ground_truth_dir      = 'mAP/input/ground-truth/'
    images_dir            = 'mAP/input/images-optional/'

    capture = cv2.VideoCapture(video_dir)
    gt_bboxes = read_xml_annotations('annotations/')
    
    list_gt_bboxes = []
    for i in range(len(gt_bboxes)):
    	list_gt_bboxes.append(gt_bboxes[str(i)])

    frame_idx = 0
    
    while True:
        success, frame = capture.read()
        if not success:
            break
        
        gt_boxes     = list_gt_bboxes[frame_idx]
        frame_idx   += 1
        labels_file  = os.path.join(labels_dir, str(frame_idx)+'.pickle') 

        with open(labels_file, "rb") as fp:
            labels = pickle.load(fp)

        bounding_boxes, scores, classes = labels         

        save_detections_for_map(frame, frame_idx, bounding_boxes, scores, gt_boxes)

        frame = draw_boxes(frame, bounding_boxes, scores, classes, min_confidence )
        frame = draw_boxes_gt(frame, gt_boxes)

        cv2.imshow('output', frame)




        cv2.waitKey(10)


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    demo(FLAGS.video_dir, FLAGS.labels_dir)
