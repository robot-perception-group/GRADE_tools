import cv2
import numpy as np
from pycocotools import mask

class Instances(object):
    '''
    The mapping dictionary of the `Instances` Class remains the same among all frames for each experiment.
    It will list all instance objects in this experiment, even though they don't appear in this frame.  
    '''
    def __init__(self, mapping, object_classes, output_img_size, NOISY_FLAG, allow_40_plus):
        self.imgsz = output_img_size
        self.object_classes = object_classes
        self.noisy_flag = NOISY_FLAG
        self.allow_40_plus =  allow_40_plus
        self.mapping = mapping
        self.annotations_obj = {
            "images": [],
            "annotations": [],
            "categories": [{"name": "person", "id": 1, "supercategory": "person"}],} #todo allow more cats
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


    def load_mask(self, instances, blur_fn=None):
        '''
        Generate mask matrix for each instance for Mask-RCNN
        '''
        classes = [] # empty detections
        bboxes = []
        semantic_mask = np.zeros([self.imgsz[1], self.imgsz[0], 1], dtype=np.uint8)
        
        for index, obj_name in enumerate(self.object_ids):
            # merge several components into one object
            masks = np.zeros(instances[0].shape, dtype=np.uint8)
            
            ids = self.object_ids[obj_name]
            for idx in ids:
                masks[instances[0]==idx] = 255
            
            # Transform mask for blurry images
            if masks.any() and self.noisy_flag:
                blur = np.load(blur_fn).item()
                masks = self.blur_mask(masks, blur)
            
            # objects exist in this image
            if masks.any():     
                # resize the full size image mask
                masks = cv2.resize(masks, dsize=self.imgsz)
                
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
            
        return semantic_mask, classes, bboxes

    def blur_mask(self, masks, blur):
        H_mean = blur['H_mean']

        masks = cv2.resize(masks, dsize=(640, 480))
        masks = cv2.warpPerspective(masks, H_mean, (640, 480), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS, borderMode=cv2.BORDER_REPLICATE)

        # add rolling shutter effect
        t_readout = blur['readout_time']
        extrinsic_mats = blur['extrinsic_mats']
        H_last = extrinsic_mats[-1,:]
        H_last = H_last.reshape((3,3))
        K = blur['intrinsic_mat']
        
        piece_H = int(1)
        y = piece_H  # y-1 is the row index

        new_pieces = []
        while y <= 480:
            # time and approximated rotation for y th row
            t_y = t_readout * y / 480
            H_y = self.interp_rot(extrinsic_mats, blur['interval'], blur['num_pose'], t_y)
            H_new = np.matmul(H_y, np.linalg.inv(H_last))
            W_y = np.matmul(np.matmul(K, H_new), np.linalg.inv(K))

            old_piece = masks[y-piece_H:y, :]
            new_piece = cv2.warpPerspective(old_piece, W_y, (640, piece_H), flags=cv2.INTER_NEAREST+cv2.WARP_FILL_OUTLIERS, borderMode=cv2.BORDER_REPLICATE)
            new_pieces.append(new_piece)
            y += piece_H

        mask_blur_rs = np.concatenate(np.array(new_pieces), axis=0)

        return mask_blur_rs
    
    def interp_rot(self, extrinsic_mats, interval, num_pose, t):
        h_array = extrinsic_mats
        exposure_ts= np.array([i * interval for i in range(num_pose+1)])
        if t >= exposure_ts[-1]:
            H_last = h_array[-1, :]
            return H_last.reshape((3,3))

        rot_t = np.array([0.]*9)
        for i in range(9):
            rot_t[i] = np.interp(t, exposure_ts, h_array[:, i])

        return rot_t.reshape((3, 3))
    
    
    def generate_mask_data(self, data_ids, OBJ_FLAG, masks, bboxes):
        if OBJ_FLAG == True:
            data_id = data_ids['obj_id']
            img_anno = {
                "id": data_id,
                "width": int(masks.shape[1]),
                "height": int(masks.shape[2]),
                "file_name": f"{data_id}.png",
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

                instance_anno = {
                    "id": instance_id,
                    "image_id": data_id,
                    "category_id": 1, # todo use data['class'] to map
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
                "file_name": f"{data_id}.png",
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