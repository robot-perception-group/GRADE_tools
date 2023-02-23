import os
import cv2
import json
import random
import shutil
import numpy as np
from PIL import Image, ImageDraw

from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Define the original dataset
main_paths = ['/home/cxu/Datasets/', '/home/cxu/Datasets2/', '/home/cxu/Datasets3/', '/home/cxu/Datasets4/',
              '/home/cxu/Datasets_new/', '/home/cxu/Datasets_new2/', '/home/cxu/Datasets_new3/',]

# output formulated dataset
train_folder = '/home/cxu/GRADE_FULL_DATASET/train/'
val_folder = '/home/cxu/GRADE_FULL_DATASET/valid/'
test_folder = '/home/cxu/GRADE_FULL_DATASET/test/'

# sub directories
image_path = 'images/'
image_blur_path = 'images_blur/'
label_path = 'labels/'
mask_path = 'masks/'

perc_dataset = 1.0 # percentage of the full dataset size

# Initialize the number of the data
max_num = 0
bg_num = 0
for main_path in main_paths:
    num_non_obj = len(os.listdir(os.path.join(main_path,'non_object/images_blur')))
    num_obj = len(os.listdir(os.path.join(main_path,'object/images_blur')))
    max_num += (num_non_obj + num_obj)
    bg_num += num_non_obj

# Define the number for split output dataset
train_num_obj = 0.8 * perc_dataset * (max_num-bg_num)
val_num_obj = 0.2 * perc_dataset * (max_num-bg_num)
test_num_obj = 0 * perc_dataset * (max_num-bg_num)

train_num_Nobj = 0.8 * perc_dataset * (bg_num)
val_num_Nobj = 0.2 * perc_dataset * (bg_num)
test_num_Nobj = 0 * perc_dataset * (bg_num)


files_object = []
for main_path in main_paths:
    objects = os.listdir(os.path.join(main_path, 'object', image_blur_path))
    for file in objects:
        files_object.append(os.path.join(main_path, 'object', image_blur_path, file))
        
random.shuffle(files_object)

files_non_object = []
for main_path in main_paths:
    non_objects = os.listdir(os.path.join(main_path, 'non_object', image_blur_path))
    for file in non_objects:
        files_non_object.append(os.path.join(main_path, 'non_object', image_blur_path, file))

random.shuffle(files_non_object)

# Define the mapping dict
mappings_obj = {}
mappings_non_obj = {}
for main_path in main_paths:
    folder_id = main_path.split('/')[-2]
    mappings_obj[folder_id] = {}
    mappings_non_obj[folder_id] = {}
    files = os.listdir(main_path)
    for file in files:
        if '_mapping.txt' in file:
            with open(os.path.join(main_path, file), 'r') as f:
                data = f.readlines()
            for i in range(len(data)):
                if 'non_object' in data[i]:
                    mappings_non_obj[folder_id][data[i].split('/')[-1][:-5]] = file.split('_mapping')[0]
                else:
                    mappings_obj[folder_id][data[i].split('/')[-1][:-5]] = file.split('_mapping')[0]
                

ids = [i for i in range(len(files_non_object) + len(files_object))]
random.shuffle(ids)

f1 = open("train_object.txt", "w")
f2 = open("valid_object.txt", "w")
f3 = open("test_object.txt", "w")

for idx, file in enumerate(files_object):
    img_id = str(file.split('/')[-1][:-4])
    folder_id = file.split('/')[3]
    exp_id = mappings_obj[folder_id][img_id]
    # Load Mask json files
    mask_fn = os.path.join('/home/cxu',folder_id,'object/masks',exp_id+'_annos_gt.json')
    import skimage.io as io
    import matplotlib.pyplot as plt
    print(folder_id, exp_id, img_id)
    
    coco = COCO(mask_fn)
    img = coco.loadImgs(int(img_id))[0] # Load images with specified ids.
    i = io.imread(file)
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=[1], iscrowd=None)
    anns = coco.loadAnns(annIds)
    plt.imshow(i); plt.axis('off')
    coco.showAnns(anns)
    plt.show()

    # Name Lable files
    label = file.replace('images_blur','labels')

    
    # if idx < train_num_obj:
    #     shutil.copyfile(file,  os.path.join(
    #         train_folder, 'images/', str(ids[idx]) + '.png'))
    #     shutil.copyfile(img_blur, os.path.join(
    #         train_folder, 'images_blur/', str(ids[idx]) + '.png'))
    #     shutil.copyfile(mask,  os.path.join(
    #         train_folder, 'masks/', str(ids[idx]) + '.npy'))
    #     shutil.copyfile(label, os.path.join(
    #         train_folder, 'labels/', str(ids[idx]) + '.txt'))
    #     f1.write('%d %s\n' % (ids[idx], file))
    # elif idx < max_number - bg_num - test_num_obj:
    #     shutil.copyfile(file,  os.path.join(
    #         val_folder, 'images/', str(ids[idx]) + '.png'))
    #     shutil.copyfile(img_blur, os.path.join(
    #         val_folder, 'images_blur/', str(ids[idx]) + '.png'))
    #     shutil.copyfile(mask,  os.path.join(
    #         val_folder, 'masks/', str(ids[idx]) + '.npy'))
    #     shutil.copyfile(label, os.path.join(
    #         val_folder, 'labels/', str(ids[idx]) + '.txt'))
    #     f2.write('%d %s\n' % (ids[idx], file))
    # elif idx < max_number - bg_num:
    #     shutil.copyfile(file, os.path.join(
    #         test_folder, 'images/', str(ids[idx]) + '.png'))
    #     shutil.copyfile(img_blur, os.path.join(
    #         test_folder, 'images_blur/', str(ids[idx]) + '.png'))
    #     shutil.copyfile(mask, os.path.join(
    #         test_folder, 'masks/', str(ids[idx]) + '.npy'))
    #     shutil.copyfile(label, os.path.join(
    #         test_folder, 'labels/', str(ids[idx]) + '.txt'))
    #     f3.write('%d %s\n' % (ids[idx], file))

f1.close()
f2.close()
f3.close()

# f1 = open("train_non_object.txt", "w")
# f2 = open("valid_non_object.txt", "w")
# f3 = open("test_non_object.txt", "w")

# for idx, file in enumerate(files_non_object):
#     img_id = str(file.split('/')[-1][:-4])
#     img_blur = os.path.join(main_path_non_object,
#                             'images_blur',  img_id + '.png')
#     label = os.path.join(main_path_non_object, 'labels', img_id + '.txt')

#     if idx < train_num_Nobj:
#         shutil.copyfile(file,  os.path.join(
#             train_folder, 'images/', str(ids[idx + len(files_object)])+'.png'))
#         shutil.copyfile(label, os.path.join(
#             train_folder, 'labels/', str(ids[idx + len(files_object)])+'.txt'))
#         shutil.copyfile(img_blur, os.path.join(
#             train_folder, 'images_blur/', str(ids[idx + len(files_object)]) + '.png'))
#         f1.write('%d %s\n' % (ids[idx+len(files_object)], file))
#     elif idx < val_num_Nobj+train_num_Nobj:
#         shutil.copyfile(file,  os.path.join(
#             val_folder, 'images/', str(ids[idx + len(files_object)])+'.png'))
#         shutil.copyfile(img_blur, os.path.join(
#             val_folder, 'images_blur/', str(ids[idx + len(files_object)]) + '.png'))
#         shutil.copyfile(label, os.path.join(
#             val_folder, 'labels/', str(ids[idx + len(files_object)])+'.txt'))
#         f2.write('%d %s\n' % (ids[idx+len(files_object)], file))
#     elif idx < val_num_Nobj+train_num_Nobj+test_num_Nobj:
#         shutil.copyfile(file, os.path.join(
#             test_folder, 'images/', str(ids[idx + len(files_object)])+'.png'))
#         shutil.copyfile(img_blur, os.path.join(
#             test_folder, 'images_blur/', str(ids[idx + len(files_object)]) + '.png'))
#         shutil.copyfile(label, os.path.join(
#             test_folder, 'labels/', str(ids[idx + len(files_object)])+'.txt'))
#         f3.write('%d %s\n' % (ids[idx+len(files_object)], file))

# f1.close()
# f2.close()
# f3.close()