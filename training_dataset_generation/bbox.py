import os
import random
import colorsys
import numpy as np
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