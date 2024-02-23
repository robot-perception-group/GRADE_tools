import cv2
import numpy as np
from pycocotools import mask

from blur import Blur

class Instances(object):
    '''
    The mapping dictionary of the `Instances` Class remains the same among all frames for each experiment.
    It will list all instance objects in this experiment, even though they don't appear in this frame.  
    '''
    def __init__(self, mapping, object_classes, output_img_size, NOISY_FLAG, allow_40_plus, KP_FLAG):
        self.imgsz = output_img_size
        self.object_classes = object_classes
        self.noisy_flag = NOISY_FLAG
        self.allow_40_plus =  allow_40_plus
        self.mapping = mapping
        categories = [{"name": "person", "id": 1, "supercategory": "person"},
                           {'supercategory': 'animal', 'id': 23, 'name': 'zebra'},
                           {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}]
        self.categories_id_map = {}
        for c in categories:
            self.categories_id_map[c['name']] = c['id']
        self.categories_supercat_map = {}
        for c in categories:
            self.categories_supercat_map[c['name']] = c['supercategory']

        # todo check categories +
        self.annotations_obj = {
            "images": [],
            "annotations": [],
            "categories": categories,
                  }
        self.annotations_non_obj = {
            "images": [],
            "annotations": [],
            "categories": categories,}
        if KP_FLAG:
            # zebra
            for id, cat in enumerate(self.annotations_obj['categories']):
                if cat['id'] == 23:
                    self.annotations_obj['categories'][id]['keypoints'] = [
                        'left_back_paw','left_back_knee','left_back_thigh',
                        'right_back_paw','right_back_knee','right_back_thigh',
                        'right_front_paw','right_front_knee','right_front_thigh',
                        'left_front_paw','left_front_knee', 'left_front_thigh',
                        'tail_end','tail_base',
                        'right_ear_tip','right_ear_base','left_ear_tip','left_ear_base',
                        'right_eye','left_eye','nose',
                        'neck_start','neck_end','skull','body_middle',
                        'back_end','back_front'
                    ]
                    self.annotations_obj['categories'][id]['skeleton'] = [
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
        
    def convert_instance(self, instances):
        '''
        1) convert the object ID from GRADE framework to NYU40 format using mapping.pkl
        2) save all object IDs in OBJECT_CLASSES
        3) return labels can not be mapped
        '''
        self.instance_dict = instances[1]
        self.object_ids = {}
        self.labels = []
        
        wrong_labels = []
    
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
                if label.lower() == 'human' or label.lower == 'clothes' or label.lower() == 'zebra' or label.lower()== 'robot':
                    obj_name = name.split('/')[1]
                else:
                    obj_name = name.split('/')[-1]

                if obj_name not in self.object_ids:
                    self.object_ids[obj_name] = [] # One object may have multiple components
                    self.labels.append(label.lower())
                self.object_ids[obj_name].append(idx)
        
        return wrong_labels


    def load_mask(self, instances, blur=None, kps=None):
        '''
        Generate mask matrix for each instance for Mask-RCNN
        '''
        classes = [] # empty detections
        bboxes = []
        semantic_mask = np.zeros([self.imgsz[1], self.imgsz[0], 1], dtype=np.uint8)
        new_kps = {}
        if kps == None:
            new_kps = None

        img_size_org = instances[0].shape


        for index, obj_name in enumerate(self.object_ids):
            # merge several components into one object
            masks = np.zeros(img_size_org, dtype=np.uint8)
            
            ids = self.object_ids[obj_name]
            for idx in ids:
                masks[instances[0]==idx] = 255
            
            # resize masks to output image size
            masks = cv2.resize(masks, dsize=(self.imgsz[0], self.imgsz[1]))
            
            # Transform mask for blurry images
            if masks.any() and self.noisy_flag and blur != None:
                masks = blur.blur_mask(masks)
            
            # objects exist in this image
            if masks.any():
                # Filter object with very small area
                rows = np.any(masks, axis=1)
                cols = np.any(masks, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                obj_height = abs(rmax - rmin) + 1
                obj_width  = abs(cmax - cmin) + 1
                
                if obj_height/obj_width > 100 or obj_width/obj_height > 100 or obj_width < 3 or obj_height < 3:
                    continue
                
                classes.append(self.labels[index])
                bboxes.append([-1, obj_name, classes[-1], -1, -1, -1, cmin, rmin, cmax, rmax])
                
                # merge multi channels semantic mask
                masks = masks[:,:,None] # add one dimension
                if len(classes) == 1:
                    semantic_mask = masks
                else:
                    semantic_mask = np.concatenate((semantic_mask, masks),axis=2)
            
                if kps and obj_name in kps.keys():
                    tmp = []
                    for k in kps[obj_name]:
                        scaled_kp = kps[obj_name][k]
                        scaled_kp[0] = int(scaled_kp[0] * self.imgsz[0] / img_size_org[1])
                        scaled_kp[1] = int(scaled_kp[1] * self.imgsz[1] / img_size_org[0])
                        tmp.extend(scaled_kp)
                    new_kps[semantic_mask.shape[2]-1] = tmp
    
        return semantic_mask, classes, bboxes, new_kps

    def generate_mask_data(self, data_ids, OBJ_FLAG, masks, bboxes, classnames, kps=None):
        if OBJ_FLAG == True:
            data_id = data_ids['obj_id']
            img_anno = {
                "id": data_id,
                "width": int(masks.shape[1]),
                "height": int(masks.shape[2]),
                "file_name": f"{data_id}.jpg",
            }
            self.annotations_obj["images"].append(img_anno)
            
            for val in range(masks.shape[-1]):
                # get binary mask
                bin_mask = masks[:, :, val].astype(bool).astype(np.uint8)
                instance_id = data_id * 100 + (val + 1)  # create id for instance, increment val

                # find bounding box
                bbox = bboxes[val]
                x, y, w, h = int(bbox[6]), int(bbox[7]), int(bbox[8])-int(bbox[6]),int(bbox[9])-int(bbox[7]) 
                
                # encode mask
                encode_mask = mask.encode(np.asfortranarray(bin_mask))
                encode_mask["counts"] = encode_mask["counts"].decode("ascii")
                size = int(mask.area(encode_mask))
                if classnames[val] == 'human':
                    classnames[val] = 'person'
                elif classnames[val] == 'robot':
                    classnames[val] = 'airplane'
                instance_anno = {
                    "id": instance_id,
                    "image_id": data_id,
                    # "category_id": self.categories_id_map[classnames[val]], # todo use data['class'] to map. check index
                    "segmentation": encode_mask,
                    "area": size,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0,
                }
                if kps is not None:
                    if val in kps.keys():
                        instance_anno["keypoints"] = kps[val]
                        instance_anno["num_keypoints"] = int(np.sum(np.array(kps[val][2::3]) > 0))

                self.annotations_obj["annotations"].append(instance_anno)
        else:
            data_id = data_ids['non_obj_id']
            img_anno = {
                "id": data_id,
                "width": int(masks.shape[1]),
                "height": int(masks.shape[2]),
                "file_name": f"{data_id}.jpg",
            }
            self.annotations_non_obj["images"].append(img_anno)
