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

from utils import read_xml_annotations

sys.path.append('Mask_RCNN')

from Mask_RCNN.export_model import export
from Mask_RCNN.config import Config
from Mask_RCNN import visualize
from Mask_RCNN import utils
import Mask_RCNN.model as modellib

import skimage.color
import skimage.io

import neptune
context = neptune.Context()
context.integrate_with_tensorflow()


parser   = argparse.ArgumentParser()
parser.add_argument('--trained_model_path', type=str, default='',help='')
parser.add_argument('--save_dir', type=str, default='',help='')
parser.add_argument('--min_confidence', type=float, default=0.9,help='')
parser.add_argument('--frames_dir', type=str, default='../datasets/AICity_data/train/S03/c010/vdo.avi',help='')

# Root directory of the project
ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# DATASET_DIR = './frames/'
# ANNOTATIONS_DIR = './annotations/'
DATASET_DIR = './data/frames/'
ANNOTATIONS_DIR = './data/annotations/'

# Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_MODEL_PATH = 'mask_rcnn_coco.h5'
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class CarsConfig(Config):
    """Configuration for training on the car shapes dataset.
    Derives from the base Config class and overrides values specific
    to the cars.
    """
    # Give the configuration a recognizable name
    NAME = "cars"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # IMAGE_MIN_DIM = 128
    # IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    # STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    # VALIDATION_STEPS = 5
    
config = CarsConfig()
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

class CarsDataset(utils.Dataset):

    def load_data(self, dataset_dir, height, width):
        # Add classes
        self.add_class("cars", 1, "car")

        bboxes = read_xml_annotations(ANNOTATIONS_DIR)
        # Add images
        frame = 0
        for filename in sorted(os.listdir(dataset_dir)):
            self.add_image("cars", image_id=frame, path=dataset_dir +'/'+filename,
                            width=width, height=height,
                            shapes=bboxes[str(frame)]
                            )
            frame +=1
    
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        info = self.image_info[image_id]
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        image = cv2.resize(image, (info['height'], info['width']))
        return image

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([1080, 1920, count], dtype=np.uint8)
        for i, (_, x1, y1, width, height, a) in enumerate(info['shapes']):
            mask[y1:y1+width, x1:x1+width, i] = 1
            image = self.load_image(image_id)
            image = cv2.resize(image, (1920, 1080))
            image = cv2.rectangle(image, (x1, y1), (x1 + width,y1 + height), (255,0,0), 10)
            print (image.shape)
            print (mask.shape)
            cv2.imshow("image", image)
            cv2.waitKey()
        # Map class names to class IDs.
        mask = cv2.resize(mask, (info['height'], info['width']))
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask, class_ids.astype(np.int32)

# Training dataset
dataset_train = CarsDataset()
dataset_train.load_data(DATASET_DIR +'train', config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = CarsDataset()
dataset_val.load_data(DATASET_DIR +'test', config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    print (class_ids)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2, 
            layers="all")


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
