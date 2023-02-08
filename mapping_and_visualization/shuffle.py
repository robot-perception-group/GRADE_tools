import os
import cv2
import random
import shutil
import numpy as np
from PIL import Image, ImageDraw

main_path_object = '/home/cxu/Datasets/object/'
main_path_non_object = '/home/cxu/Datasets/non_object/'

# output
train_folder = '/home/cxu/GRADE_DATASET/train/'
val_folder = '/home/cxu/GRADE_DATASET/valid/'
test_folder = '/home/cxu/GRADE_DATASET/test/'

# sub directories
image_path = 'images/'
image_blur_path = 'images_blur/'
label_path = 'labels/'
mask_path = 'masks/'

max_number = 20000
bg_num = 0.1 * max_number
train_num_obj = 0.8 * (max_number-bg_num)
val_num_obj = 0.1 * (max_number-bg_num)
test_num_obj = 0.1 * (max_number-bg_num)

train_num_Nobj = 0.8 * (bg_num)
val_num_Nobj = 0.1 * (bg_num)
test_num_Nobj = 0.1 * (bg_num)


files_object = os.listdir(os.path.join(main_path_object, image_path))
for idx, file in enumerate(files_object):
    files_object[idx] = os.path.join(main_path_object, image_path, file)

random.shuffle(files_object)


files_non_object = os.listdir(os.path.join(main_path_non_object, image_path))
for idx, file in enumerate(files_non_object):
    files_non_object[idx] = os.path.join(main_path_non_object, image_path, file)

random.shuffle(files_non_object)

ids = [i for i in range(len(files_non_object) + len(files_object))]
random.shuffle(ids)

f1 = open("train_object.txt", "w")
f2 = open("valid_object.txt", "w")
f3 = open("test_object.txt", "w")

for idx, file in enumerate(files_object):
    img_id = str(file.split('/')[-1][:-4])
    img_blur = os.path.join(
        main_path_object, image_blur_path,  img_id + '.png')
    mask = os.path.join(main_path_object, mask_path,  img_id + '.npy')
    label = os.path.join(main_path_object, label_path, img_id + '.txt')

    if idx < train_num_obj:
        shutil.copyfile(file,  os.path.join(
            train_folder, 'images/', str(ids[idx]) + '.png'))
        shutil.copyfile(img_blur, os.path.join(
            train_folder, 'images_blur/', str(ids[idx]) + '.png'))
        shutil.copyfile(mask,  os.path.join(
            train_folder, 'masks/', str(ids[idx]) + '.npy'))
        shutil.copyfile(label, os.path.join(
            train_folder, 'labels/', str(ids[idx]) + '.txt'))
        f1.write('%d %s\n' % (ids[idx], file))
    elif idx < max_number - bg_num - test_num_obj:
        shutil.copyfile(file,  os.path.join(
            val_folder, 'images/', str(ids[idx]) + '.png'))
        shutil.copyfile(img_blur, os.path.join(
            val_folder, 'images_blur/', str(ids[idx]) + '.png'))
        shutil.copyfile(mask,  os.path.join(
            val_folder, 'masks/', str(ids[idx]) + '.npy'))
        shutil.copyfile(label, os.path.join(
            val_folder, 'labels/', str(ids[idx]) + '.txt'))
        f2.write('%d %s\n' % (ids[idx], file))
    elif idx < max_number - bg_num:
        shutil.copyfile(file, os.path.join(
            test_folder, 'images/', str(ids[idx]) + '.png'))
        shutil.copyfile(img_blur, os.path.join(
            test_folder, 'images_blur/', str(ids[idx]) + '.png'))
        shutil.copyfile(mask, os.path.join(
            test_folder, 'masks/', str(ids[idx]) + '.npy'))
        shutil.copyfile(label, os.path.join(
            test_folder, 'labels/', str(ids[idx]) + '.txt'))
        f3.write('%d %s\n' % (ids[idx], file))

f1.close()
f2.close()
f3.close()

f1 = open("train_non_object.txt", "w")
f2 = open("valid_non_object.txt", "w")
f3 = open("test_non_object.txt", "w")

for idx, file in enumerate(files_non_object):
    img_id = str(file.split('/')[-1][:-4])
    img_blur = os.path.join(main_path_non_object,
                            'images_blur',  img_id + '.png')
    label = os.path.join(main_path_non_object, 'labels', img_id + '.txt')

    if idx < train_num_Nobj:
        shutil.copyfile(file,  os.path.join(
            train_folder, 'images/', str(ids[idx + len(files_object)])+'.png'))
        shutil.copyfile(label, os.path.join(
            train_folder, 'labels/', str(ids[idx + len(files_object)])+'.txt'))
        shutil.copyfile(img_blur, os.path.join(
            train_folder, 'images_blur/', str(ids[idx + len(files_object)]) + '.png'))
        f1.write('%d %s\n' % (ids[idx+len(files_object)], file))
    elif idx < val_num_Nobj+train_num_Nobj:
        shutil.copyfile(file,  os.path.join(
            val_folder, 'images/', str(ids[idx + len(files_object)])+'.png'))
        shutil.copyfile(img_blur, os.path.join(
            val_folder, 'images_blur/', str(ids[idx + len(files_object)]) + '.png'))
        shutil.copyfile(label, os.path.join(
            val_folder, 'labels/', str(ids[idx + len(files_object)])+'.txt'))
        f2.write('%d %s\n' % (ids[idx+len(files_object)], file))
    elif idx < val_num_Nobj+train_num_Nobj+test_num_Nobj:
        shutil.copyfile(file, os.path.join(
            test_folder, 'images/', str(ids[idx + len(files_object)])+'.png'))
        shutil.copyfile(img_blur, os.path.join(
            test_folder, 'images_blur/', str(ids[idx + len(files_object)]) + '.png'))
        shutil.copyfile(label, os.path.join(
            test_folder, 'labels/', str(ids[idx + len(files_object)])+'.txt'))
        f3.write('%d %s\n' % (ids[idx+len(files_object)], file))

f1.close()
f2.close()
f3.close()