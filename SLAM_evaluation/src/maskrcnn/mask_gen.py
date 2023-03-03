import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


from samples.coco import coco


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.DETECTION_NMS_THRESHOLD = 0.5
config.RPN_NMS_THRESHOLD = 0.5
# config.MAX_GT_INSTANCES = 50
# config.BACKBONE = "resnet50"
# config.IMAGE_MIN_DIM = 256
# config.IMAGE_MAX_DIM = 256
# config.GPU_COUNT = 1
# config.IMAGES_PER_GPU = 1
# config.TRAIN_ROIS_PER_IMAGE = 50
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
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

# Load a random image from the images folder
IMAGE_DIR = sys.argv[1]
SEMANTIC_DIR = sys.argv[2]

if not os.path.exists(SEMANTIC_DIR):
    os.makedirs(SEMANTIC_DIR)

file_names = next(os.walk(IMAGE_DIR))[2]
file_names.sort(key=lambda x:int(x[:-4]))


import cv2
for file_name in file_names:
    images = []
    images.append(cv2.imread(IMAGE_DIR + file_name))
    print('Loading %s' %(file_name),end='\r')
    
    # Run detection
    results = model.detect(images, verbose=0)

    # Visualize results
    r = results[0]['masks']
    mask = np.zeros((r.shape[0],r.shape[1]))

    for i in range(r.shape[2]):
        index = np.where(r[:,:,i]==True)
        pixel_list = list(zip(index[0],index[1]))
        for x,y in pixel_list:
            mask[x,y] = i+1
    
    fn = str(int(file_name[:-4])-1).zfill(6) + '.txt' # zfill can be removed
    # Save the semantic results
    np.savetxt(SEMANTIC_DIR + fn, mask, fmt='%d', delimiter=' ')
    
    img_fn = os.path.join(SEMANTIC_DIR, fn[:-4] +'.png')
    # save the images
    visualize.display_instances(img_fn, images[0], results[0]['rois'], results[0]['masks'], results[0]['class_ids'], class_names, results[0]['scores'])
