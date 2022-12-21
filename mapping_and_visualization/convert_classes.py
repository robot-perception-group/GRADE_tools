import numpy as np
import pickle as pkl
import cv2
import random
from PIL import Image, ImageDraw

def random_colours(N, enable_random=True, num_channels=3):
    """
    Generate random colors.
    Generate visually distinct colours by linearly spacing the hue
    channel in HSV space and then convert to RGB space.
    """
    start = 0
    if enable_random:
        random.seed(10)
        start = random.random()
    hues = [(start + i / N) % 1.0 for i in range(N)]
    colours = [list(colorsys.hsv_to_rgb(h, 0.9, 1.0)) for i, h in enumerate(hues)]
    if num_channels == 4:
        for color in colours:
            color.append(1.0)
    if enable_random:
        random.shuffle(colours)
    return colours

def colorize_bboxes(bboxes_2d_data, bboxes_2d_rgb, num_channels=3):
    """ Colorizes 2D bounding box data for visualization.


        Args:
            bboxes_2d_data (numpy.ndarray): 2D bounding box data from the sensor.
            bboxes_2d_rgb (numpy.ndarray): RGB data from the sensor to embed bounding box.
            num_channels (int): Specify number of channels i.e. 3 or 4.
    """
    semantic_id_list = []
    bbox_2d_list = []
    rgb_img = Image.fromarray(bboxes_2d_rgb)
    rgb_img_draw = ImageDraw.Draw(rgb_img)
    for bbox_2d in bboxes_2d_data:
        if bbox_2d[5] > 0:
            semantic_id_list.append(bbox_2d[1])
            bbox_2d_list.append(bbox_2d)
    semantic_id_list_np = np.unique(np.array(semantic_id_list))
    color_list = random_colours(len(semantic_id_list_np.tolist()), True, num_channels)
    for bbox_2d in bbox_2d_list:
        index = np.where(semantic_id_list_np == bbox_2d[1])[0][0]
        bbox_color = color_list[index]
        outline = (int(255 * bbox_color[0]), int(255 * bbox_color[1]), int(255 * bbox_color[2]))
        if num_channels == 4:
            outline = (
                int(255 * bbox_color[0]),
                int(255 * bbox_color[1]),
                int(255 * bbox_color[2]),
                int(255 * bbox_color[3]),
            )
        rgb_img_draw.rectangle([(bbox_2d[6], bbox_2d[7]), (bbox_2d[8], bbox_2d[9])], outline=outline, width=2)
    bboxes_2d_rgb = np.array(rgb_img)
    return bboxes_2d_rgb

def detect_occlusion(rgb, depth, seg, depth_thr):
    rgb_mask = np.zeros(rgb.shape, dtype=np.uint8)
    depth_mask = np.zeros(rgb.shape, dtype=np.uint8)
    seg_mask = np.zeros(rgb.shape, dtype=np.uint8)
    rgb_mask[np.where((rgb <= [15,15,15]).all(axis=2))] = [255,255,255]
    depth_mask[depth < depth_thr] = [255,255,255]
    seg_mask[seg >= 40] = [255,255,255] # assuming 40 are flying objects/humans
    perc_rgb = (np.count_nonzero(rgb_mask)/ (3 * rgb.shape[0] * rgb.shape[1])) * 100
    perc_depth = (np.count_nonzero(depth_mask)/ (3 * rgb.shape[0] * rgb.shape[1])) * 100
    perc_seg = (np.count_nonzero(seg_mask)/ (3 * rgb.shape[0] * rgb.shape[1])) * 100
    return perc_rgb, perc_depth, perc_seg

def convert_instances(instances, mapping):
    for idx, name, label in zip(instances[1]['uniqueId'],instances[1]['name'],instances[1]['semanticLabel']):
        if allow_40_plus and "human" == label and "body" not in name:
            label = "clothes"
    index, = np.where(instances[1]['uniqueId'] == idx)
    try:
        instances[1]['semanticId'][index[0]] = mapping[label.lower()]
        instances[0][instances[0] == instances[1]['uniqueId'][index[0]]] = mapping[label.lower()]
    except:
        import ipdb; ipdb.set_trace()
    return instances

allow_40_plus = False
additional = {'robot':41, 'flying-object':42, 'clothes':43}
#mapping = pkl.load(open('~/Desktop/mapping.pkl','rb'))
if allow_40_plus:
    mapping = {**mapping,**additional}

import os
main_path = '' # This will process all the subfolders recursively
dirs = os.listdir(main_path)
wrong_rgb = {}
wrong_depth = {}
for d in dirs:
    print(f"processing {d}")
    mapping = pkl.load(open('mapping.pkl','rb'))
    rgb_path = os.path.join(main_path, d, 'Viewport0_occluded', 'rgb')
    depth_path = os.path.join(main_path, d, 'Viewport0_occluded', 'depthLinear')
    instance_path = os.path.join(main_path, d, 'Viewport0_occluded', 'instance')
    wrong_rgb[d] = []
    wrong_depth[d] = []
    for i in range(1,1800):
        print(f"{i}/1800", end='\r')
        rgb = cv2.imread(os.path.join(rgb_path, f'{i}.png'))
        depth = np.load(os.path.join(depth_path, f'{i}.npy'))
        instances = np.load(os.path.join(instance_path, f'{i}.npy'), allow_pickle = True)
        instances = convert_instances(instances, mapping)
        a,b,c = detect_occlusion(rgb, depth, instances[0], 0.05)
        if a > 10:
            wrong_rgb[d].append(i)
        if b > 10:
            wrong_depth[d].append(i)
    import ipdb; ipdb.set_trace()


## todo for bboxes -- humans use only /my_human_* as box. the rest is big
## use colorize with rgb to vis