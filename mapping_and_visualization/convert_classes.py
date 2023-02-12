import os
import cv2
import json
import random
import shutil
import confuse
import colorsys
import argparse
import numpy as np
import pickle as pkl
from PIL import Image, ImageDraw
from pycocotools import mask

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

def detect_occlusion(rgb, depth, depth_thr): # TO DO: add seg
    rgb_mask = np.zeros(rgb.shape, dtype=np.uint8)
    depth_mask = np.zeros(rgb.shape, dtype=np.uint8)
    #seg_mask = np.zeros(rgb.shape, dtype=np.uint8)
    
    rgb_mask[np.where((rgb <= [15,15,15]).all(axis=2))] = [255,255,255]
    depth_mask[depth < depth_thr] = [255,255,255]
    #seg_mask[seg >= 40] = [255,255,255] # assuming 40 are flying objects/humans
    
    # calculate the percentage of the rgb / depth are occluded
    perc_rgb = (np.count_nonzero(rgb_mask) / (3 * rgb.shape[0] * rgb.shape[1])) * 100
    perc_depth = (np.count_nonzero(depth_mask) / (3 * rgb.shape[0] * rgb.shape[1])) * 100
    #perc_seg = (np.count_nonzero(seg_mask) / (3 * rgb.shape[0] * rgb.shape[1])) * 100
    
    return perc_rgb, perc_depth


class Instances(object):
    '''
    For each experiment, the mapping dictionary of `Instances` Class remains the same.
    '''
    def __init__(self, mapping, object_classes, output_img_size, allow_40_plus):
        self.imgsz = output_img_size
        self.object_classes = object_classes
        self.allow_40_plus =  allow_40_plus
        self.mapping = mapping
        self.annotations_obj = {
            "images": [],
            "annotations": [],
            "categories": [{"name": "person", "id": 1, "supercategory": "person"}],}
        self.annotations_non_obj = {
            "images": [],
            "annotations": [],
            "categories": [{"name": "person", "id": 1, "supercategory": "person"}],}
        
    def convert_instance(self, instances, wrong_labels):
        self.instance_dict = instances[1]
        self.object_ids = {}
        self.labels = []
    
        for idx, name, label in zip(self.instance_dict['uniqueId'], self.instance_dict['name'], self.instance_dict['semanticLabel']):
            if self.allow_40_plus and "human" == label and "body" not in name:
                label = "clothes"
        
            # Transform the label ID to NYU 40
            try:
                obj_ID = self.mapping[label.lower()]
                self.instance_dict[idx-1]['semanticId'] = obj_ID
            except:
                print(label, 'can not be mapped...')
                print(idx, name, label)
                wrong_labels.append(label)        
        
            if label.lower() in self.object_classes:
                if label.lower() == 'human':
                    obj_name = name.split('/')[1]
                else:
                    obj_name = name.split('/')[-1]

                if obj_name not in self.object_ids:
                    self.object_ids[obj_name] = [] # One object may have multiple components
                    self.labels.append(label.lower())
                self.object_ids[obj_name].append(idx)
        
        return wrong_labels


    def load_mask(self, instances):
        '''
        Specific Mask Generation for MASK RCNN
        '''
        classes = [] # empty detections
        semantic_mask = np.zeros([self.imgsz[1], self.imgsz[0], 1], dtype=np.uint8)
        
        for index, obj_name in enumerate(self.object_ids):
            # merge several components into one object
            masks = np.zeros(instances[0].shape, dtype=np.uint8)
            
            ids = self.object_ids[obj_name]
            for idx in ids:
                masks[instances[0]==idx] = 255
            
            # objects exist in this image
            if masks.any():     
                # resize the full size image mask
                masks = cv2.resize(masks, dsize=self.imgsz)
                masks = masks[:,:,None] # add one dimension
                
                # Filter object with very small area
                obj_height = np.max(np.where(masks > 0)[0]) - np.min(np.where(masks > 0)[0]) + 1
                obj_width  = np.max(np.where(masks > 0)[1]) - np.min(np.where(masks > 0)[1]) + 1
                
                if obj_height/obj_width > 100 or obj_width/obj_height > 100 or obj_width < 3 or obj_height < 3:
                    continue
                
                classes.append(self.labels[index])
                
                # merge multi channels semantic mask
                if len(classes) == 1:
                    semantic_mask = masks
                else:
                    semantic_mask = np.concatenate((semantic_mask, masks),axis=2)
            
        return semantic_mask, classes


    def generate_mask_data(self, data_ids, OBJ_FLAG, masks):
        if OBJ_FLAG == True:
            data_id = data_ids['obj_id']
            img_anno = {
                "id": data_id,
                "width": int(masks.shape[1]),
                "height": int(masks.shape[2]),
                "file_name": "{}.png".format(data_id),
            }
            self.annotations_obj["images"].append(img_anno)
            
            for val in range(masks.shape[-1]):
                # get binary mask
                bin_mask = masks[:, :, val].astype(bool).astype(np.uint8)
                instance_id = data_id * 100 + (val + 1)  # create id for instance, increment val
                # find bounding box
                def bbox2(img):
                    rows = np.any(img, axis=1)
                    cols = np.any(img, axis=0)
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    return int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)

                # encode mask
                encode_mask = mask.encode(np.asfortranarray(bin_mask))
                encode_mask["counts"] = encode_mask["counts"].decode("ascii")
                size = int(mask.area(encode_mask))
                x, y, w, h = bbox2(bin_mask)

                instance_anno = {
                    "id": instance_id,
                    "image_id": data_id,
                    "category_id": 1, # use data['class'] to map
                    "segmentation": encode_mask,
                    "area": size,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0,
                }

                self.annotations_obj["annotations"].append(instance_anno)
        else:
            data_id = data_ids['non_obj_id']
            img_anno = {
                "id": data_id,
                "width": int(masks.shape[1]),
                "height": int(masks.shape[2]),
                "file_name": "{}.png".format(data_id),
            }
            self.annotations_non_obj["images"].append(img_anno)
            
            bin_mask = masks[:, :, 0].astype(bool).astype(np.uint8)
            instance_id = data_id * 100  # create id for instance, increment val

            # encode mask
            encode_mask = mask.encode(np.asfortranarray(bin_mask))
            encode_mask["counts"] = encode_mask["counts"].decode("ascii")
            size = 0

            instance_anno = {
                "id": instance_id,
                "image_id": data_id,
                "category_id": 1, # use data['class'] to map
                "segmentation": encode_mask,
                "area": size,
                "bbox": [0, 0, 0, 0],
                "iscrowd": 0,
            }

            self.annotations_non_obj["annotations"].append(instance_anno)

class Bboxes(object):
    def __init__(self, mapping, object_classes, filtered_classes, input_img_size, output_img_size, allow_40_plus):
        self.input_imgsz = input_img_size
        self.output_imgsz = output_img_size
        self.object_classes = object_classes
        self.filtered_classes = filtered_classes
        self.allow_40_plus =  allow_40_plus
        self.mapping = mapping   
    

    def load_bbox(self, bbox_2d_data):
        """ Resize RGB image and corresponding coordinates of bbox rectangle.

            Args:
                rgb (numpy.ndarray): RGB data from the sensor to embed bounding box.
                bboxes_2d_data (numpy.ndarray): 2D bounding box data from the sensor.
                input_img_size(tuple): Desired output image size
        """
        new_bbox_data = []
        img_width = self.input_imgsz[0]
        img_height = self.input_imgsz[1]
        
        for bbox in bbox_2d_data:
            obj_class = bbox[2].lower() 
            obj_name  = bbox[1]
            
            if obj_class in self.object_classes and 'Armature' not in obj_name:
                bbox[6] = bbox[6] / img_width  * self.output_imgsz[0]
                bbox[7] = bbox[7] / img_height * self.output_imgsz[1]
                bbox[8] = bbox[8] / img_width  * self.output_imgsz[0]
                bbox[9] = bbox[9] / img_height * self.output_imgsz[1]
                
                # Filter based on the area
                bbox_width = abs(bbox[8] - bbox[6]) + 1
                bbox_height = abs(bbox[9] - bbox[7]) + 1
                
                if bbox_height/bbox_width > 100 or bbox_width/bbox_height > 100 or bbox_width < 3 or bbox_height < 3:
                    continue
                
                new_bbox_data.append(bbox)
            
        new_bbox_data = np.array(new_bbox_data)
                
        return new_bbox_data


    def convert_bbox(self, bbox_2d_data, wrong_labels):
        """ Transform the semantic label ID of bbox_2d_data

        Args:
            bboxes_2d_data (numpy.ndarray): 2D bounding box data from the sensor.
        """
        for bbox in bbox_2d_data:
            obj_class = bbox[2].lower()
            obj_name  = bbox[1]
            
            # Filter out class labels
            if obj_class in self.filtered_classes or 'Armature' in obj_name:
                continue
            
            # transform semantic ID to the mapping ID based on the semantic label
            try:
                obj_ID = self.mapping[obj_class]
                bbox[5] = obj_ID
            except:
                print(obj_class, 'can not be mapped...')
                wrong_labels.append(obj_class)
        
        return bbox_2d_data, wrong_labels
    
    @staticmethod
    def colorize_bboxes(bboxes_2d_data, rgb, num_channels=3):
        """ Colorizes 2D bounding box data for visualization.


            Args:
                bboxes_2d_data (numpy.ndarray): 2D bounding box data from the sensor.
                rgb (numpy.ndarray): RGB data from the sensor to embed bounding box.
                num_channels (int): Specify number of channels i.e. 3 or 4.
        """
        obj_name_list = []
        rgb_img = Image.fromarray(rgb)
        rgb_img_draw = ImageDraw.Draw(rgb_img)
        
        for bbox_2d in bboxes_2d_data:
            obj_name_list.append(bbox_2d[1])

        obj_name_list_np = np.unique(np.array(obj_name_list))
        color_list = random_colours(len(obj_name_list_np.tolist()), True, num_channels)
        
        for bbox_2d in bboxes_2d_data:
            index = np.where(obj_name_list_np == bbox_2d[1])[0][0]
            bbox_color = color_list[index]
            outline = (int(255 * bbox_color[0]), int(255 * bbox_color[1]), int(255 * bbox_color[2]))
            if num_channels == 4:
                outline = (
                    int(255 * bbox_color[0]),
                    int(255 * bbox_color[1]),
                    int(255 * bbox_color[2]),
                    int(255 * bbox_color[3]),
                )
            rgb_img_draw.rectangle([(bbox_2d[6], bbox_2d[7]), (bbox_2d[8], bbox_2d[9])], outline=outline, width=3)
            bboxes_2d_rgb = np.array(rgb_img)
        return bboxes_2d_rgb


    def generate_bbox_data(self, output_path, data_ids, data_type, bboxes_data):
        if data_type:
            folder = 'object'
            data_id = data_ids['obj_id']
        else:
            folder = 'non_object'
            data_id = data_ids['non_obj_id']
        
        img_width = self.output_imgsz[0]
        img_height = self.output_imgsz[1]
        
        f1 = open(os.path.join(output_path, folder, 'labels', f"{data_id}.txt"), "w")
        for bbox in bboxes_data:
            obj_label = bbox[2]
            if obj_label.lower() in self.object_classes:
                index = self.object_classes.index(obj_label.lower())
                width    = (bbox[8] - bbox[6]) / img_width
                height   = (bbox[9] - bbox[7]) / img_height
                x_center = (bbox[8] + bbox[6]) / (2 * img_width)
                y_center = (bbox[9] + bbox[7]) / (2 * img_height)
                f1.write('%d %.6f %.6f %.6f %.6f\n' %(index, x_center, y_center, width, height))
        f1.close()


def main(config):
    # Define input variables
    main_paths = config['main_path'].get()
    output_path = config['output_path'].get()
    viewport = config['viewport_name'].get()
    
    # Create output directories
    output_paths = [os.path.join(output_path, 'object'),
                    os.path.join(output_path, 'object','masks'),
                    os.path.join(output_path, 'object','labels'),
                    os.path.join(output_path, 'object','images'),
                    os.path.join(output_path, 'object','images_blur'),
                    os.path.join(output_path, 'non_object'),
                    os.path.join(output_path, 'non_object','masks'),
                    os.path.join(output_path, 'non_object','labels'),
                    os.path.join(output_path, 'non_object','images'),
                    os.path.join(output_path, 'non_object','images_blur'),]
    
    for path in output_paths:
        if not os.path.exists(path):
            os.makedirs(path)

    # Define the ignored classes during mapping
    filtered_classes = config['filtered_classes'].get()
    object_classes = config['object_classes'].get()
    
    # Define image size
    input_img_size = config['input_image_size'].get()
    output_img_size = config['output_image_size'].get()
    
    # Load mapping dictionary
    mapping = pkl.load(open(config['mapping_file'].get(),'rb'))
    allow_40_plus = config['allow_40_plus'].get()
    if allow_40_plus:
        additional = config['additional'].get()
        mapping = {**mapping, **additional}
    
    # initialize data file id
    data_ids = {}
    data_ids['obj_id'] = 0 # data with desired objects (positive data)
    data_ids['non_obj_id'] = 0 # data with no object (background/negative data)
    
    OBJ_FLAG = None
    INSTANCE_FLAG = config['instance'].get()
    BBOX_FLAG = config['bbox'].get()

    # Transform Data into desired dataset
    f1 = open(os.path.join(output_path, "wrong_labels.txt"), "w")
    
    for path in main_paths:
        # list all experiments in one of the main path
        dirs = os.listdir(path)
        exp_n = path.split('/')[-2] # experiment name, eg: DE_cam0, DE_cam1
        for d in dirs:             
            print(f"processing {path}{d}")
            rgb_path = os.path.join(path, d, viewport, 'rgb')
            rgb_blur_path = os.path.join('/home/cxu',exp_n, d, viewport, 'rgb')
            depth_path = os.path.join(path, d, viewport, 'depthLinear')
            instance_path = os.path.join(path, d, viewport, 'instance')
            bbox_path = os.path.join(path, d, viewport, 'bbox_2d_tight')

            # Check repository
            sub_dirs = [not os.path.exists(sub_d) for sub_d in [rgb_path, depth_path, instance_path, bbox_path, rgb_blur_path]]
            if np.any(sub_dirs):
                print(d, ' HAVE INCOMPLETE DATA...')
                continue
            
            # initialize ignored data list
            wrong_labels = []
            f2 = open(os.path.join(output_path,f"{d}_ignored_ids.txt"), "w")
            f3 = open(os.path.join(output_path,f"{d}_mapping.txt"), "w")

            # initial the instance mapping dictionary
            if INSTANCE_FLAG:
                instance = Instances(mapping, object_classes, output_img_size, allow_40_plus)

                instances = np.load(os.path.join(instance_path, f'{1}.npy'), allow_pickle = True)
                wrong_labels = instance.convert_instance(instances, wrong_labels)
                
                if wrong_labels != []:
                    for label in wrong_labels:
                        f1.write('%s : %s \n' %(d, label))
            
            # initial the bbox mapping dictionary 
            if BBOX_FLAG:
                bbox = Bboxes(mapping, object_classes, filtered_classes, input_img_size, output_img_size, allow_40_plus)
            
            for i in range(1,1802):
                print(f"{i}/1801")
                # check if all data exist
                rgb_fn = os.path.join(rgb_path, f'{i}.png')
                rgb_blur_fn = os.path.join(rgb_blur_path, f'{i}.png')
                depth_fn = os.path.join(depth_path, f'{i}.npy')
                instance_fn = os.path.join(instance_path, f'{i}.npy')
                bbox_fn = os.path.join(bbox_path, f'{i}.npy')
                
                fns = [not os.path.exists(fn) for fn in [rgb_fn, depth_fn, instance_fn, bbox_fn]]
                if np.any(fns):
                    continue
                
                # load rgb and depth image
                rgb = cv2.imread(rgb_fn)
                depth = np.load(depth_fn)

                # Detect Occlusion
                perc_rgb_occluded, perc_depth_occluded = detect_occlusion(rgb, depth, 0.3)
                if perc_rgb_occluded > 10 and perc_depth_occluded > 10:
                    f2.write('%d\n' %(i))
                    continue
                    
                if perc_depth_occluded > 40:
                    f2.write('%d\n' %(i))
                    continue
                
                # resize image to output format size
                # rgb_resized = cv2.resize(rgb, dsize=(output_img_size[0], output_img_size[1]))
                # rgb_ = rgb_resized.copy()
                
                # Load instance
                if INSTANCE_FLAG:
                    instances =  np.load(instance_fn, allow_pickle = True)
                    masks, classes = instance.load_mask(instances)  # generate mask and detected classes
                    
                    if len(classes) == 0:
                        OBJ_FLAG = False # negative sample
                        data_ids['non_obj_id'] += 1
                    else:
                        OBJ_FLAG = True # postive sample
                        data_ids['obj_id'] += 1
                        
                    # save mask data
                    instance.generate_mask_data(data_ids, OBJ_FLAG, masks)
                    
                    # # # visualiza semantic mask
                    # for j in range(mask.shape[2]):
                    #     rgb_[np.where((mask[:,:,j] > 0))] = [255,255,255]
                        
                # Load bboxes
                if BBOX_FLAG:
                    bboxes = np.load(bbox_fn, allow_pickle = True)
            
                    # Transform bboxes label ID to NYU40
                    # bboxes, wrong_labels = bbox.convert_bbox(bboxes, wrong_labels)
                    filtered_bboxes = bbox.load_bbox(bboxes)
                    
                    # when only processing bbox data
                    if not INSTANCE_FLAG:
                        if filtered_bboxes.shape[0] == 0:
                            OBJ_FLAG = False # negative sample
                            data_ids['non_obj_id'] += 1
                        else:
                            OBJ_FLAG = True # postive sample
                            data_ids['obj_id'] += 1
                            
                    # save bbox data
                    bbox.generate_bbox_data(output_path, data_ids, OBJ_FLAG, filtered_bboxes)
                    
                    # bbox visualization
                    # rgb_ = bbox.colorize_bboxes(filtered_bboxes, rgb_)
                
                
                # Save RGB images
                if OBJ_FLAG:
                    folder = 'object'
                    data_id = data_ids['obj_id']
                else:
                    folder = 'non_object'
                    data_id = data_ids['non_obj_id']
                
                # write the rgb and rgb_static images
                # fn = os.path.join(output_path, folder, 'images', f"{data_id}.png")
                # cv2.imwrite(fn, rgb_resized)
                
                # write the blur images
                blur_fn = os.path.join(output_path, folder, 'images_blur', f"{data_id}.jpg")
                
                if not os.path.exists(rgb_blur_fn):
                    rgb_resized = cv2.resize(rgb, dsize=(output_img_size[0], output_img_size[1]))
                    cv2.imwrite(blur_fn, rgb_resized)
                else:
                    blur_img = cv2.imread(rgb_blur_fn)
                    cv2.imwrite(blur_fn, blur_img)
                    #shutil.copyfile(rgb_blur_fn, blur_fn)
                
                # Record the mapping relations
                f3.write('%s -> %s\n' %(os.path.join(d, f'{i}.png'), os.path.join(folder, f"{data_id}.jpg")))
                print(os.path.join(d, f'{i}.png'), " -> ", os.path.join(folder, f"{data_id}.jpg"))
                
                del instances, bboxes, masks
                del rgb, depth
                
                # # visualize the image result
                # cv2.imshow('bbox + mask', rgb_)
                # cv2.waitKey(1)
            
            # Close the files for mapping relations and ignored ids
            f2.close()
            f3.close()
            
            # Save the semantic mask json data
            anno_path_object = os.path.join(output_path, 'object','masks', f"{d}_annos_gt.json")
            anno_path_non_object = os.path.join(output_path, 'non_object','masks', f"{d}_annos_gt.json")
            json.dump(instance.annotations_obj, open(anno_path_object, "w+"))
            json.dump(instance.annotations_non_obj, open(anno_path_non_object, "w+"))
            
    f1.close()



if __name__ == '__main__':
    # Define parser arguments
    parser = argparse.ArgumentParser(description="Dataset Generation")
    parser.add_argument("--config", type=str, default="mapping_and_visualization/convert_classes.yaml", help="Path to Config File")
    args, _ = parser.parse_known_args()
    
    # load configuration file
    config = confuse.Configuration("Data Generation", __name__)
    config.set_file(args.config)
    config.set_args(args)
    
    main(config)