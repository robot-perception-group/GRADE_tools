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

random.seed(42)

# Define the original dataset
# todo change folders
parent_path = '/media/ebonetto/WindowsData/superclose_DS3/proc'
exp_names = ["Windmills_populated"]
# output formulated dataset
out_path = '/media/ebonetto/WindowsData/superclose_DS3/DS'

main_paths = [os.path.join(parent_path, exp_name) for exp_name in exp_names]
has_subviewport = True

categories = [{"name": "person", "id": 1, "supercategory": "person"},
              {'supercategory': 'animal', 'id': 23, 'name': 'zebra', 'keypoints': [
                        'left_back_paw','left_back_knee','left_back_thigh',
                        'right_back_paw','right_back_knee','right_back_thigh',
                        'right_front_paw','right_front_knee','right_front_thigh',
                        'left_front_paw','left_front_knee', 'left_front_thigh',
                        'tail_end','tail_base',
                        'right_ear_tip','right_ear_base','left_ear_tip','left_ear_base',
                        'right_eye','left_eye','nose',
                        'neck_start','neck_end','skull','body_middle',
                        'back_end','back_front'],
                    'skeleton': [
                                [1, 2], [2, 3], [3, 26],
                                [4, 5], [5, 6], [6, 26],
                                [7, 8], [8, 9],
                                [10,11], [11,12],
                                [13, 14],
                                [15, 16], [17,18],
                                [16, 19], [19, 20], [18, 20],
                                [19, 21], [20, 21], [19, 24], [20, 24],
                                [21, 24], [24, 23], [23, 22], [22, 27], [27, 9], [27, 12], [27, 25], [25, 26], [26, 14]
                                ]
                     },
           {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}]

annots_train = {
    "images": [],
    "annotations": [],
    "categories": categories,}

annots_valid = {
    "images": [],
    "annotations": [],
    "categories": categories}

train_folder = os.path.join(out_path, 'train/')
val_folder = os.path.join(out_path, 'val/')
test_folder = os.path.join(out_path, 'test/')
for path in [train_folder, val_folder, test_folder]:
    if not os.path.exists(path):
        os.makedirs(path)

# sub directories
is_blur = False
image_path =  f'images{"" if not is_blur else "_blur"}/'
label_path = 'labels/'
mask_path = 'masks/'
for path in [image_path, label_path, mask_path]:
    for folder in [train_folder, val_folder, test_folder]:
        if not os.path.exists(os.path.join(folder, path)):
            os.makedirs(os.path.join(folder, path))

perc_dataset = 1.0 # percentage of the full dataset size

# Initialize the number of the data
max_num = 0
bg_num = 0
for main_path in main_paths:
    if has_subviewport:
        viewports = os.listdir(os.path.join(main_path))
        for viewport in viewports:
            num_non_obj = len(os.listdir(os.path.join(main_path, viewport, 'non_object/images')))
            num_obj = len(os.listdir(os.path.join(main_path, viewport, 'object/images')))
            max_num += (num_non_obj + num_obj)
            bg_num += num_non_obj
    else:
        num_non_obj = len(os.listdir(os.path.join(main_path, 'non_object/images')))
        num_obj = len(os.listdir(os.path.join(main_path, 'object/images')))
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
    if has_subviewport:
        viewports = os.listdir(main_path)
        for viewport in viewports:
            objects = os.listdir(os.path.join(main_path, viewport, 'object', image_path))
            for file in objects:
                files_object.append(os.path.join(main_path, viewport, 'object', image_path, file))
    else:
        objects = os.listdir(os.path.join(main_path, 'object', image_path))
        for file in objects:
            files_object.append(os.path.join(main_path, 'object', image_path, file))

files_non_object = []
for main_path in main_paths:
    if has_subviewport:
        viewports = os.listdir(main_path)
        for viewport in viewports:
            non_objects = os.listdir(os.path.join(main_path, viewport,  'non_object', image_path))
            for file in non_objects:
                files_non_object.append(os.path.join(main_path, viewport, 'non_object', image_path, file))
    else:
        non_objects = os.listdir(os.path.join(main_path, 'non_object', image_path))
        for file in non_objects:
            files_non_object.append(os.path.join(main_path, 'non_object', image_path, file))

random.shuffle(files_non_object)
random.shuffle(files_object)

# Define the mapping dict
mappings_obj = {}
mappings_non_obj = {}
for exp_name in exp_names:
    folder_id = exp_name
    idx = exp_names.index(exp_name)
    main_path = main_paths[idx]
    mappings_obj[folder_id] = {}
    mappings_non_obj[folder_id] = {}
    if has_subviewport:
        viewports = os.listdir(main_path)
        for viewport in viewports:
            files = os.listdir(os.path.join(main_path, viewport))
            for file in files:
                if '_mapping.txt' in file:
                    with open(os.path.join(main_path, viewport, file), 'r') as f:
                        data = f.readlines()
                    for i in range(len(data)):
                        if 'non_object' in data[i]:
                            mappings_non_obj[folder_id][data[i].split('/')[-1][:-5]] = [viewport, data[i].split('.jpg')[0].split('/')[-1]]
                        else:
                            mappings_obj[folder_id][data[i].split('/')[-1][:-5]] = [viewport, data[i].split('.jpg')[0].split('/')[-1]]
    else:
        files = os.listdir(main_path)
        for file in files:
            if '_mapping.txt' in file:
                with open(os.path.join(main_path, file), 'r') as f:
                    data = f.readlines()
                for i in range(len(data)):
                    if 'non_object' in data[i]:
                        mappings_non_obj[folder_id][data[i].split('/')[-1][:-5]] = ['', data[i].split('.jpg')[0].split('/')[-1]]
                    else:
                        mappings_obj[folder_id][data[i].split('/')[-1][:-5]] = ['', data[i].split('.jpg')[0].split('/')[-1]]


ids = [i for i in range(len(files_non_object) + len(files_object))]
random.shuffle(ids)

f1 = open(os.path.join(out_path,f"train_object{'' if not is_blur else '_blur'}.txt"), "w")
f2 = open(os.path.join(out_path,f"valid_object{'' if not is_blur else '_blur'}.txt"), "w")
f3 = open(os.path.join(out_path,f"test_object{'' if not is_blur else '_blur'}.txt"), "w")

id_counter = 0

for idx, file in enumerate(files_object):
    print(f'{idx+1}/{len(files_object)}')
    img_id = str(file.split('/')[-1][:-4])
    folder_id = file[len(os.path.join(parent_path)) + 1:].split('/')[0]
    exp_id = mappings_obj[folder_id][img_id][0]
    exp_num = mappings_obj[folder_id][img_id][1]


    # Load Mask json files
    mask_fn = os.path.join(parent_path, folder_id, exp_id,'object/masks',folder_id+'_annos_gt.json')

    with open(mask_fn) as f:
        masks = json.load(f)
        annots = masks['annotations']

    # import skimage.io as io
    # import matplotlib.pyplot as plt
    # print(folder_id, exp_id, img_id)
    #
    # coco = COCO(mask_fn)
    # img = coco.loadImgs(int(img_id))[0] # Load images with specified ids.
    # i = io.imread(file)
    # annIds = coco.getAnnIds(imgIds=img['id'], catIds=[24], iscrowd=None)
    # anns = coco.loadAnns(annIds)
    # plt.imshow(i); plt.axis('off')
    # coco.showAnns(anns)
    # plt.show()

    # Name Lable files
    label = file.replace('images','labels').replace('jpg','txt')


    if idx < train_num_obj:
        shutil.copyfile(file,  os.path.join(
            train_folder, 'images/', str(ids[idx]) + '.jpg'))
        shutil.copyfile(label, os.path.join(
            train_folder, 'labels/', str(ids[idx]) + '.txt'))
        f1.write('%d  %s/%s.jpg  %s\n' % (ids[idx], exp_id, exp_num, file))

        img_anno = {
            "id": ids[idx],
            "width": int(1920),
            "height": int(1080),
            "file_name": "{}.jpg".format(ids[idx]),}
        annots_train["images"].append(img_anno)

        for n, annot in enumerate(annots):
            if annot['image_id'] == int(img_id):
                # instance_id = ids[idx] * 100 + (n + 1)
                instance_id = id_counter
                id_counter += 1
                annot['image_id'] = ids[idx]
                annot['id'] = instance_id
                annots_train["annotations"].append(annot)
        del masks, annots

    elif idx < max_num - bg_num - test_num_obj:
        shutil.copyfile(file,  os.path.join(
            val_folder, 'images/', str(ids[idx]) + '.jpg'))
        shutil.copyfile(label, os.path.join(
            val_folder, 'labels/', str(ids[idx]) + '.txt'))
        f2.write('%d  %s/%s.jpg  %s\n' % (ids[idx], exp_id, exp_num, file))

        img_anno = {
            "id": ids[idx],
            "width": int(1920),
            "height": int(1080),
            "file_name": "{}.jpg".format(ids[idx]),}
        annots_valid["images"].append(img_anno)

        for n, annot in enumerate(annots):
            if annot['image_id'] == int(img_id):
                # instance_id = ids[idx] * 100 + (n + 1)
                instance_id = id_counter
                id_counter += 1
                annot['image_id'] = ids[idx]
                annot['id'] = instance_id
                annots_valid["annotations"].append(annot)
        del masks, annots

f1.close()
f2.close()
f3.close()

f1 = open(os.path.join(out_path,f"train_non_object{'' if not is_blur else '_blur'}.txt"), "w")
f2 = open(os.path.join(out_path,f"valid_non_object{'' if not is_blur else '_blur'}.txt"), "w")
f3 = open(os.path.join(out_path,f"test_non_object{'' if not is_blur else '_blur'}.txt"), "w")

for idx, file in enumerate(files_non_object):
    print(f'{idx+1}/{len(files_non_object)}')
    img_id = str(file.split('/')[-1][:-4])
    folder_id = file[len(os.path.join(parent_path)) + 1:].split('/')[0]

    exp_id = mappings_non_obj[folder_id][img_id][0]
    exp_num = mappings_non_obj[folder_id][img_id][1]
    label = file.replace('images','labels').replace('jpg','txt')

    new_img_id = ids[idx + len(files_object)]
    if idx < train_num_Nobj:
        shutil.copyfile(file,  os.path.join(
            train_folder, 'images/', str(new_img_id)+'.jpg'))
        shutil.copyfile(label, os.path.join(
            train_folder, 'labels/', str(new_img_id)+'.txt'))
        f1.write('%d  %s/%s  %s\n' % (new_img_id, exp_id, exp_num, file))

        img_anno = {
            "id": new_img_id,
            "width": int(1920),
            "height": int(1080),
            "file_name": "{}.jpg".format(new_img_id),}
        annots_train["images"].append(img_anno)

        # bin_mask = np.zeros([480,1920], dtype=np.uint8).astype(bool)
        # # instance_id = new_img_id * 100  # create id for instance, increment val
        # instance_id = id_counter
        # id_counter += 1

        # # encode mask
        # encode_mask = mask.encode(np.asfortranarray(bin_mask))
        # encode_mask["counts"] = encode_mask["counts"].decode("ascii")
        # size = 0

        # annot = {
        #     "id": instance_id,
        #     "image_id": new_img_id,
        #     "category_id": 1, # use data['class'] to map
        #     "segmentation": encode_mask,
        #     "area": size,
        #     "bbox": [0, 0, 0, 0],
        #     "iscrowd": 0,
        # }
        # annots_train["annotations"].append(annot)

    elif idx < val_num_Nobj+train_num_Nobj:
        shutil.copyfile(file,  os.path.join(
            val_folder, 'images/', str(new_img_id)+'.jpg'))
        shutil.copyfile(label, os.path.join(
            val_folder, 'labels/', str(new_img_id)+'.txt'))
        f2.write('%d  %s/%s  %s\n' % (new_img_id, exp_id, exp_num, file))

        img_anno = {
            "id": new_img_id,
            "width": int(1920),
            "height": int(1080),
            "file_name": "{}.jpg".format(new_img_id),}
        annots_valid["images"].append(img_anno)

f1.close()
f2.close()
f3.close()

anno_path_train = os.path.join(train_folder, "masks", "train_annos_gt.json")
anno_path_valid = os.path.join(val_folder, "masks", "val_annos_gt.json")
json.dump(annots_train, open(anno_path_train, "w+"))
json.dump(annots_valid, open(anno_path_valid, "w+"))
