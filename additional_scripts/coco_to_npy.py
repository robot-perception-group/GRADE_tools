import os
import sys
import random
import math
import re
import time
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import Mask RCNN
import utils
import visualize
import model as modellib
from model import log
from dataset import GRADEDataset
from dataset import GRADEConfig
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
from coco import evaluate_coco
from coco import CocoDataset

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

dataset_val = CocoDataset()
coco = dataset_val.load_coco("/home/ebonetto/coco", "val", year=2017, return_coco=True, class_ids=[1], auto_download=False)
dataset_val.prepare()

image_ids = dataset_val.image_ids

for img in range(len(imgIds)):
	anns = ct.getAnnIds(imgIds=imgIds[img],catIds=[1])
	anns = ct.loadAnns(anns)
	info = ct.loadImgs(imgIds[img])[0]

	masks = np.zeros(shape=(height,width, len(anns)))
	for ann in range(len(anns)):
		masks[:,:,ann] += ct.annToMask(anns[ann])*255
	masks = masks.astype(np.uint8)
	d = {'mask': masks,'class': ['human']*len(anns)}
	np.save(f"/home/ebonetto/coco/val2017/masks/{fname}",d)