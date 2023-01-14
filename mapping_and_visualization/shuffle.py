import os
import cv2
import random
import shutil
import numpy as np
from PIL import Image, ImageDraw

# main_path  = '/home/cxu/Datasets2/object/'
# main_path_ = '/home/cxu/Datasets3/object/'

# image_path = 'images/'
# label_path = 'labels/'
# mask_path  = 'masks/'

# files = os.listdir(main_path + image_path)
# for idx, file in enumerate(files):
#     files[idx] = main_path + image_path + file

# files_ = os.listdir(main_path_ + image_path)
# for idx, file in enumerate(files_):
#     files_[idx] = main_path_ + image_path +  file
    
# files = files + files_
# #files.sort(key=lambda x:int(x[:-4]))

# random.shuffle(files)

# f1 = open("train_object.txt", "w")
# f2 = open("valid_object.txt", "w")
# f3 = open("test_object.txt", "w")

# for idx, file in enumerate(files):
#     folder = file.split('/')[3]
#     img_id = str(file.split('/')[-1][:-4])
#     mask  = os.path.join('/home/cxu', folder, 'object/masks',  img_id + '.npy')
#     label = os.path.join('/home/cxu', folder, 'object/labels', img_id + '.txt')
    
#     if idx < 16200:
#         shutil.copyfile(file,  '/home/cxu/GRADE_DATASET/train/images/' + str(2*idx+1)+'.png')
#         shutil.copyfile(mask,  '/home/cxu/GRADE_DATASET/train/masks/'  + str(2*idx+1)+'.npy')
#         shutil.copyfile(label, '/home/cxu/GRADE_DATASET/train/labels/' + str(2*idx+1)+'.txt')
#         f1.write('%d %s\n' %(2*idx+1, file))
#     elif idx < 17100:
#         shutil.copyfile(file,  '/home/cxu/GRADE_DATASET/valid/images/' + str(2*idx+1)+'.png')
#         shutil.copyfile(mask,  '/home/cxu/GRADE_DATASET/valid/masks/'  + str(2*idx+1)+'.npy')
#         shutil.copyfile(label, '/home/cxu/GRADE_DATASET/valid/labels/' + str(2*idx+1)+'.txt')
#         f2.write('%d %s\n' %(2*idx+1, file))
#     elif idx < 18000:
#         shutil.copyfile(file,  '/home/cxu/GRADE_DATASET/test/images/' + str(2*idx+1)+'.png')
#         shutil.copyfile(mask,  '/home/cxu/GRADE_DATASET/test/masks/'  + str(2*idx+1)+'.npy')
#         shutil.copyfile(label, '/home/cxu/GRADE_DATASET/test/labels/' + str(2*idx+1)+'.txt')
#         f3.write('%d %s\n' %(2*idx+1, file))
        
# f1.close()
# f2.close()
# f3.close()


main_path  = '/home/cxu/Datasets2/non_object/'
main_path_ = '/home/cxu/Datasets3/non_object/'

image_path = 'images/'

files = os.listdir(main_path + image_path)
for idx, file in enumerate(files):
    files[idx] = main_path + image_path + file

files_ = os.listdir(main_path_ + image_path)
for idx, file in enumerate(files_):
    files_[idx] = main_path_ + image_path +  file
    
files = files + files_
#files.sort(key=lambda x:int(x[:-4]))

random.shuffle(files)

f1 = open("train_non_object.txt", "w")
f2 = open("valid_non_object.txt", "w")
f3 = open("test_non_object.txt", "w")

for idx, file in enumerate(files):
    folder = file.split('/')[3]
    img_id = str(file.split('/')[-1][:-4])
    #mask  = os.path.join('/home/cxu', folder, 'non_object/masks',  img_id + '.npy')
    label = os.path.join('/home/cxu', folder, 'non_object/labels', img_id + '.txt')
    
    if idx < 1800:
        shutil.copyfile(file,  '/home/cxu/GRADE_DATASET/train/images/' + str(2*(idx+1))+'.png')
        #shutil.copyfile(mask,  '/home/cxu/GRADE_DATASET/train/masks/'  + str(2*idx+1)+'.npy')
        shutil.copyfile(label, '/home/cxu/GRADE_DATASET/train/labels/' + str(2*(idx+1))+'.txt')
        f1.write('%d %s\n' %(2*idx+1, file))
    elif idx < 1900:
        shutil.copyfile(file,  '/home/cxu/GRADE_DATASET/valid/images/' + str(2*(idx+1))+'.png')
        #shutil.copyfile(mask,  '/home/cxu/GRADE_DATASET/valid/masks/'  + str(2*idx+1)+'.npy')
        shutil.copyfile(label, '/home/cxu/GRADE_DATASET/valid/labels/' + str(2*(idx+1))+'.txt')
        f2.write('%d %s\n' %(2*idx+1, file))
    elif idx < 2000:
        shutil.copyfile(file,  '/home/cxu/GRADE_DATASET/test/images/' + str(2*(idx+1))+'.png')
        #shutil.copyfile(mask,  '/home/cxu/GRADE_DATASET/test/masks/'  + str(2*idx+1)+'.npy')
        shutil.copyfile(label, '/home/cxu/GRADE_DATASET/test/labels/' + str(2*(idx+1))+'.txt')
        f3.write('%d %s\n' %(2*idx+1, file))
        
f1.close()
f2.close()
f3.close()

# for test in test_data:
#     test_txt = test[:-4]+'.txt'
#     shutil.move(main_path+test, main_path + 'test/')
#     shutil.move(label_path+test_txt, label_path + 'test/')


# for file in files:
#     img = cv2.imread(os.path.join(main_path, file))
#     img = Image.fromarray(img)
#     rgb_img_draw = ImageDraw.Draw(img)
#     with open(os.path.join(label_path, file[:-4]+'.txt')) as f:
#         data = f.readlines()
#         if data != []:
#             for bbox in data:
#                 x_center = float(bbox.split(' ')[1])
#                 y_center = float(bbox.split(' ')[2])
#                 width = float(bbox.split(' ')[3])
#                 height = float(bbox.split(' ')[4])
#                 x1 = (x_center - (width/2)) * 960
#                 x2 = (x_center + (width/2)) * 960
#                 y1 = (y_center - (height/2)) * 720
#                 y2 = (y_center + (height/2)) * 720
#                 rgb_img_draw.rectangle([(x1, y1), (x2, y2)],  width=2)
#     img_bbox = np.array(img)
#     cv2.imshow('img',img_bbox)
#     cv2.waitKey(0)

# for train in train_data:
#     train_txt = train[:-4]+'.txt'
#     shutil.move(main_path+train, main_path + 'train/')
#     shutil.move(label_path+train_txt, label_path + 'train/')