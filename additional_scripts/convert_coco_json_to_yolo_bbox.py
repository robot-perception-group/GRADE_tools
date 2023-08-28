import json
import os
import shutil

import requests
from tqdm import tqdm


def make_folders(path="output", img_path=''):
	if img_path == '':
		if os.path.exists(path):
			shutil.rmtree(path)
		os.makedirs(path)
	else:
		if os.path.exists(path):
			shutil.rmtree(path)
		os.makedirs(os.path.join(path, "images"))
		os.makedirs(os.path.join(path, "labels"))
	return path


def convert_bbox_coco2yolo(img_width, img_height, bbox):
	"""
	Convert bounding box from COCO  format to YOLO format

	Parameters
	----------
	img_width : int
			width of image
	img_height : int
			height of image
	bbox : list[int]
			bounding box annotation in COCO format:
			[top left x position, top left y position, width, height]

	Returns
	-------
	list[float]
			bounding box annotation in YOLO format:
			[x_center_rel, y_center_rel, width_rel, height_rel]
	"""

	# YOLO bounding box format: [x_center, y_center, width, height]
	# (float values relative to width and height of image)
	x_tl, y_tl, w, h = bbox
	if w < 0:
		x_tl += w
		w = -w
	if h < 0:
		y_tl += h
		h = -h

	if x_tl < 0:
		x_tl = 0
	if x_tl > img_width:
		x_tl = img_width
	if y_tl < 0:
		y_tl = 0
	if y_tl > img_height:
		y_tl = img_height
	if x_tl + w > img_width:
		w = img_width - x_tl
	if y_tl + h > img_height:
		h = img_height - y_tl

	dw = 1.0 / img_width
	dh = 1.0 / img_height

	x_center = x_tl + w / 2.0
	y_center = y_tl + h / 2.0

	x = x_center * dw
	y = y_center * dh
	w = w * dw
	h = h * dh

	return [x, y, w, h]


def convert_coco_json_to_yolo_txt(output_path, json_file, img_path=''):
	path = make_folders(output_path, img_path)
	print(json_file)
	with open(json_file) as f:
		json_data = json.load(f)

	# write _darknet.labels, which holds names of all classes (one class per line)
	label_file = os.path.join(path, "_darknet.labels")
	with open(label_file, "w") as f:
		for category in tqdm(json_data["categories"], desc="Categories"):
			category_name = category["name"]
			f.write(f"{category_name}\n")

	for image in tqdm(json_data["images"], desc="Annotation txt for each iamge"):
		img_id = image["id"]
		img_name = image["file_name"]
		img_width = image['width']
		img_height = image['height']
		if img_path != '':
			try:
				shutil.copy(os.path.join(img_path, img_name), os.path.join(path, "images", img_name))
			except:
				img_data = requests.get(image['coco_url']).content
				with open(os.path.join(path, "images", img_name), 'wb') as handler:
					handler.write(img_data)
		anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
		if img_path != '':
			anno_txt = os.path.join(path, "labels", img_name[:img_name.rfind(".")] + ".txt")
		else:
			anno_txt = os.path.join(path, img_name[:img_name.rfind(".")] + ".txt")
		with open(anno_txt, "w") as f:
			for anno in anno_in_image:
				category = 0 if anno['category_id'] == 24 else 0  # 4 #anno["category_id"]-1
				bbox_COCO = anno["bbox"]
				x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
				f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

	print("Converting COCO Json to YOLO txt finished!")


import sys

args = sys.argv[1:]
if len(args) == 0 or len(args) > 3:
	print("First arg out folder, second the json file. If three args, third is the image folder")
	print("When three args are used, the images are copied FROM the third arg to the output/images folder")
	sys.exit()
if len(args) > 2:
	convert_coco_json_to_yolo_txt(args[0], args[1], args[2])
else:
	convert_coco_json_to_yolo_txt(args[0], args[1])
