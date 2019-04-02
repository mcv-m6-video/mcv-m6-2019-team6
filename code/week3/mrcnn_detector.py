import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import argparse
import pickle

sys.path.append('Mask_RCNN')

from Mask_RCNN.export_model import export
from Mask_RCNN.config import Config
from Mask_RCNN import visualize
from Mask_RCNN import utils
import Mask_RCNN.model as modellib

parser   = argparse.ArgumentParser()
parser.add_argument('--trained_model_path', type=str, default='',help='')
parser.add_argument('--save_dir', type=str, default='',help='')
parser.add_argument('--min_confidence', type=float, default=0.9,help='')
parser.add_argument('--video_dir', type=str, default='../datasets/AICity_data/train/S03/c010/vdo.avi',help='')

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80
    NAME = "coco"

config = InferenceConfig()
config.display()

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def detect(trained_model_path, video_dir, save_dir, min_confidence):    
    # Recreate the model in inference mode    
    config.DETECTION_MIN_CONFIDENCE = min_confidence
    model = modellib.MaskRCNN(mode="inference", 
                              config=config,
                              model_dir=trained_model_path)
    model.load_weights(trained_model_path, by_name=True)
    print("Loading weights from ", trained_model_path)
    model.load_weights(trained_model_path, by_name=True)


    capture = cv2.VideoCapture(video_dir)

    frame_idx = 0
    
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_idx += 1
        results = model.detect([frame], verbose=1)
        # Visualize results
        r = results[0]
        #visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'], 
        #                        class_names, r['scores'])

        bounding_boxes = r['rois'].tolist()
        scores         = r['scores'].tolist()
        class_ids      = r['class_ids']
        classes        = [class_names[c] for c in class_ids]

        labels         = [bounding_boxes, scores, classes]
        labels_file    = os.path.join(save_dir, str(frame_idx)+'.pickle') 

        with open(labels_file, "wb") as fp:
            pickle.dump(labels, fp)        

    return bounding_boxes, scores





if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    detect(FLAGS.trained_model_path, FLAGS.video_dir, FLAGS.save_dir, FLAGS.min_confidence)
