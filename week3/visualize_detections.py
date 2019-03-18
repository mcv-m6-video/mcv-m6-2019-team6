import numpy as np
import cv2
import pickle
from coco_class_names import class_names
import argparse
import os
import random
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



def demo(video_dir, labels_dir):

    capture = cv2.VideoCapture(video_dir)
    frame_idx = 0

    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_idx += 1
        labels_file    = os.path.join(labels_dir, str(frame_idx)+'.pickle') 

        with open(labels_file, "rb") as fp:
            labels = pickle.load(fp)

        bounding_boxes, scores, classes = labels         
        frame = draw_boxes(frame, bounding_boxes, scores, classes, min_confidence )

        cv2.imshow('output', frame)
        cv2.waitKey(10)




if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    demo(FLAGS.video_dir, FLAGS.labels_dir)
