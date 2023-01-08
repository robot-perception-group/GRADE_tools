import os
import cv2
import random
import confuse
import colorsys
import argparse
import numpy as np
import pickle as pkl
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


    def load_mask(self, instances):
        '''
        Specific Mask Generation for MASK RCNN
        '''
        classes = [] # empty detections
        semantic_mask = np.zeros([self.imgsz[1], self.imgsz[0], 1], dtype=np.uint8)
        
        for index, obj_name in enumerate(self.object_ids):
            # merge several components into one object
            mask = np.zeros(instances[0].shape, dtype=np.uint8)
            
            ids = self.object_ids[obj_name]
            for idx in ids:
                mask[instances[0]==idx] = 255
            
            # objects exist in this image
            if mask.any():     
                # resize the full size image mask
                mask = cv2.resize(mask, dsize=self.imgsz)
                mask = mask[:,:,None] # add one dimension
                
                # Filter object with very small area
                obj_height = np.max(np.where(mask > 0)[0]) - np.min(np.where(mask > 0)[0])
                obj_width  = np.max(np.where(mask > 0)[1]) - np.min(np.where(mask > 0)[1])
                
                if obj_height/obj_width > 100 or obj_width/obj_height > 100 or obj_width < 3 or obj_height < 3:
                    continue
                
                classes.append(self.labels[index])
                
                # merge multi channels semantic mask
                if len(classes) == 1:
                    semantic_mask = mask
                else:
                    semantic_mask = np.concatenate((semantic_mask, mask),axis=2)
            
        return semantic_mask, classes
    
    @staticmethod
    def generate_mask_data(output_path, data_ids, mask, classes):
        data_id = data_ids['obj_id']
        masks = {}
        masks['mask'] = mask
        masks['class'] = classes
        np.save(os.path.join(output_path, 'object','masks', f"{data_id}.npy"), masks)

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
                bbox_width = abs(bbox[8] - bbox[6])
                bbox_height = abs(bbox[9] - bbox[7])
                
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
            if obj_class in self.filtered_class_label or 'Armature' in obj_name:
                continue
            
            # transform semantic ID to the mapping ID based on the semantic label
            try:
                obj_ID = self.mapping[obj_class]
                bbox[5] = obj_ID
            except:
                print(obj_class, 'can not be mapped...')
                wrong_labels.append(obj_class)
        
        return bbox_2d_data
    
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
                    os.path.join(output_path, 'object','images'),
                    os.path.join(output_path, 'object','labels'),
                    os.path.join(output_path, 'non_object'),
                    os.path.join(output_path, 'non_object','images'),
                    os.path.join(output_path, 'non_object','labels')]
    
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
    
    # initialize ignored data list
    wrong_img_ids = {}
    wrong_labels = [] # list with labels that can not be mapped
    
    # initialize data file id
    data_ids = {}
    data_ids['obj_id'] = 0 # data with desired objects (positive data)
    data_ids['non_obj_id'] = 0 # data with no object (background/negative data)
    
    OBJ_FLAG = None
    INSTANCE_FLAG = config['instance'].get()
    BBOX_FLAG = config['bbox'].get()
    
    # initialize data classes
    if INSTANCE_FLAG:
        instance = Instances(mapping, object_classes, output_img_size, allow_40_plus)
        
    if BBOX_FLAG:
        bbox = Bboxes(mapping, object_classes, filtered_classes, input_img_size, output_img_size, allow_40_plus)


    # Transform Data into desired dataset
    for path in main_paths:
        # list all experiments in one of the main path
        dirs = os.listdir(path)
        wrong_img_ids[path] = {}

        for d in dirs:            
            print(f"processing {path}/{d}")
            rgb_path = os.path.join(path, d, viewport, 'rgb')
            depth_path = os.path.join(path, d, viewport, 'depthLinear')
            instance_path = os.path.join(path, d, viewport, 'instance')
            bbox_path = os.path.join(path, d, viewport, 'bbox_2d_tight')
        
            wrong_img_ids[path][d] = []
            
            # initial the instance mapping dictionary
            if INSTANCE_FLAG:
                instances = np.load(os.path.join(instance_path, f'{1}.npy'), allow_pickle = True)
                instance.convert_instance(instances, wrong_labels)
            
            for i in range(1,900):
                print(f"{i}/900", end='\r')
                
                rgb = cv2.imread(os.path.join(rgb_path, f'{2*i}.png'))
                depth = np.load(os.path.join(depth_path, f'{2*i}.npy'))

                # Detect Occlusion
                perc_rgb_occluded, perc_depth_occluded = detect_occlusion(rgb, depth, 0.3)
                if perc_rgb_occluded > 10 and perc_depth_occluded > 10:
                    wrong_img_ids[path][d].append(2*i)
                    continue
                    
                if perc_depth_occluded > 40:
                    wrong_img_ids[path][d].append(2*i)
                    continue
                
                # resize image to output format size
                rgb_resized = cv2.resize(rgb, dsize=(output_img_size[0], output_img_size[1]))
                # rgb_ = rgb_resized.copy()
                
                # Load instance
                if INSTANCE_FLAG:
                    instances = np.load(os.path.join(instance_path, f'{2*i}.npy'), allow_pickle = True)
                    mask, classes = instance.load_mask(instances)  # generate mask and detected classes
                    
                    if len(classes) == 0:
                        OBJ_FLAG = False # negative sample
                        data_ids['non_obj_id'] += 1
                    else:
                        OBJ_FLAG = True # postive sample
                        data_ids['obj_id'] += 1
                        
                        # save mask data
                        instance.generate_mask_data(output_path, data_ids, mask, classes)
                    
                        # # # visualiza semantic mask
                        # for j in range(mask.shape[2]):
                        #     rgb_[np.where((mask[:,:,j] > 0))] = [255,255,255]
                        
                # Load bboxes
                if BBOX_FLAG:
                    bboxes = np.load(os.path.join(bbox_path, f'{2*i}.npy'), allow_pickle = True)
            
                    # Transform bboxes label ID to NYU40
                    # bboxes = bbox.convert_bbox(bboxes, wrong_labels)
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
                    
                    # # bbox visualization
                    # rgb_ = bbox.colorize_bboxes(filtered_bboxes, rgb_)
                
                
                # Save RGB images
                if OBJ_FLAG:
                    folder = 'object'
                    data_id = data_ids['obj_id']
                else:
                    folder = 'non_object'
                    data_id = data_ids['non_obj_id']
                fn = os.path.join(output_path, folder, 'images', f"{data_id}.png")
                cv2.imwrite(fn, rgb_resized)
                
                
                # # visualize the image result
                # cv2.imshow('bbox + mask', rgb_)
                # cv2.waitKey(1)
            
            
    np.save(os.path.join(output_path,'ignored_img_ids.npy'), wrong_img_ids)
    np.save(os.path.join(output_path,'wrong_labels.npy'), wrong_labels)



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