import os
import cv2
import random
import shutil
import numpy as np
from PIL import Image, ImageDraw

main_path = '/home/cxu/Datasets/images/train'
label_path = '/home/cxu/Datasets/labels/train'

files = os.listdir(main_path)
files.sort(key=lambda x:int(x[:-4]))

random.shuffle(files)

# for test in test_data:
#     test_txt = test[:-4]+'.txt'
#     shutil.move(main_path+test, main_path + 'test/')
#     shutil.move(label_path+test_txt, label_path + 'test/')


for file in files:
    img = cv2.imread(os.path.join(main_path, file))
    img = Image.fromarray(img)
    rgb_img_draw = ImageDraw.Draw(img)
    with open(os.path.join(label_path, file[:-4]+'.txt')) as f:
        data = f.readlines()
        if data != []:
            for bbox in data:
                x_center = float(bbox.split(' ')[1])
                y_center = float(bbox.split(' ')[2])
                width = float(bbox.split(' ')[3])
                height = float(bbox.split(' ')[4])
                x1 = (x_center - (width/2)) * 960
                x2 = (x_center + (width/2)) * 960
                y1 = (y_center - (height/2)) * 720
                y2 = (y_center + (height/2)) * 720
                rgb_img_draw.rectangle([(x1, y1), (x2, y2)],  width=2)
    img_bbox = np.array(img)
    cv2.imshow('img',img_bbox)
    cv2.waitKey(0)

# for train in train_data:
#     train_txt = train[:-4]+'.txt'
#     shutil.move(main_path+train, main_path + 'train/')
#     shutil.move(label_path+train_txt, label_path + 'train/')