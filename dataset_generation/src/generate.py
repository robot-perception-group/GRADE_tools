import os
import cv2
import json
import confuse
import argparse
import numpy as np
import pickle as pkl

from instance import Instances
from bbox import Bboxes
from blur import Blur

def detect_occlusion(rgb, depth, depth_thr): # todo add segmentation perhaps
    rgb_mask = np.zeros(rgb.shape, dtype=np.uint8)
    depth_mask = np.zeros(rgb.shape, dtype=np.uint8)
    
    rgb_mask[np.where((rgb <= [15,15,15]).all(axis=2))] = [255,255,255]
    depth_mask[depth < depth_thr] = [255,255,255]
    
    # calculate the percentage of the rgb / depth are occluded
    perc_rgb = (np.count_nonzero(rgb_mask) / (3 * rgb.shape[0] * rgb.shape[1])) * 100
    perc_depth = (np.count_nonzero(depth_mask) / (3 * rgb.shape[0] * rgb.shape[1])) * 100
    
    return perc_rgb, perc_depth

def visualize(rgb, masks):
    # masks shape: [rgb.shape[0], rgb.shape[1]]
    rgb = rgb.astype(float)

    overlay = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype="float")
    alpha_mask = np.zeros(overlay.shape)
    alpha_bg = np.ones(overlay.shape)

    overlay[np.where((masks[:,:] > 0))] = [255,255,255]
    alpha_mask[np.where((masks[:,:] > 0))] = 0.5

    foreground = cv2.multiply(alpha_mask, overlay)
    background = cv2.multiply(alpha_bg-alpha_mask, rgb)
    rgb = cv2.add(foreground[...,:3], background)

    return rgb/255

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
                    os.path.join(output_path, 'non_object'),
                    os.path.join(output_path, 'non_object','masks'),
                    os.path.join(output_path, 'non_object','labels'),
                    os.path.join(output_path, 'non_object','images'),]
    
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
    NOISY_FLAG = config['noisy'].get() # Generate Blur datasets

    # Transform Data into desired dataset
    for path in main_paths:
        # list all experiments in one of the main path
        dirs = os.listdir(path)
        exp_n = path.split('/')[-2] # experiment name, eg: DE_cam0, DE_cam1
        
        # loop through each complete experiment [60s with 1801 images]
        for d in dirs:
            if not 'd94ec' in d:
                continue             
            print(f"processing {path}{d}")
            rgb_path = os.path.join(path, d, viewport, 'rgb')
            depth_path = os.path.join(path, d, viewport, 'depthLinear')
            instance_path = os.path.join(path, d, viewport, 'instance')
            bbox_path = os.path.join(path, d, viewport, 'bbox_2d_tight')

            # Check repository
            sub_dirs = [not os.path.exists(sub_d) for sub_d in [rgb_path, depth_path, instance_path, bbox_path]]
            if np.any(sub_dirs):
                print(d, ' HAVE INCOMPLETE DATA...')
                continue
            
            if NOISY_FLAG:
                blur_path = os.path.join('/home/cxu', exp_n, d, viewport, 'blur')
                blur_path = os.path.join('/home/cxu/Test_BLUR/blur_params1')
                if not os.path.exists(blur_path):
                    print(d, ' MISSING BLUR DATA ...')
                    continue
            
            # initialize ignored data list
            f2 = open(os.path.join(output_path, f"{d}_ignored_ids.txt"), "w")
            f3 = open(os.path.join(output_path, f"{d}_mapping.txt"), "w")

            # initial the instance mapping dictionary
            if INSTANCE_FLAG:
                instance = Instances(mapping, object_classes, output_img_size, NOISY_FLAG, allow_40_plus)

                instances = np.load(os.path.join(instance_path, f'{1}.npy'), allow_pickle = True)
                wrong_labels = instance.convert_instance(instances)
                
                if wrong_labels != []:
                    f1 = open(os.path.join(output_path, f"{d}_wrong_labels.txt"), "w")
                    for label in wrong_labels:
                        f1.write('%s\n' %(label))
                    f1.close()
            
            # initial the bbox mapping dictionary 
            if BBOX_FLAG:
                bbox = Bboxes(mapping, object_classes, filtered_classes, input_img_size, output_img_size, allow_40_plus)
            
            for i in range(1,1802):
                print(f"{i}/1801")
                # check if all data exist
                rgb_fn = os.path.join(rgb_path, f'{i}.png')
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
                rgb_resized = cv2.resize(rgb, dsize=(output_img_size[0], output_img_size[1]))
                
                # Generate blur rgb image
                if NOISY_FLAG:
                    blur_fn = os.path.join(blur_path, f'{i}.npy')
                    if os.path.exists(blur_fn):
                        blur = Blur(blur_fn, output_img_size)
                        rgb_blur = blur.blur_image(rgb_resized)
                    else:
                        blur = None
                        rgb_blur = rgb_resized
                
                # Load instance
                if INSTANCE_FLAG:
                    instances =  np.load(instance_fn, allow_pickle = True)
                    
                    if NOISY_FLAG:
                        masks, classes, bboxes = instance.load_mask(instances, blur)  # generate mask and detected classes
                    else:
                        masks, classes, bboxes = instance.load_mask(instances)
                        
                    if len(classes) == 0:
                        OBJ_FLAG = False # negative sample
                        data_ids['non_obj_id'] += 1
                    else:
                        OBJ_FLAG = True # postive sample
                        data_ids['obj_id'] += 1
                        
                    # save mask data
                    instance.generate_mask_data(data_ids, OBJ_FLAG, masks, bboxes)
                    
                    # # visualiza semantic mask
                    masks_ = np.zeros((output_img_size[1],output_img_size[0]))
                    for j in range(masks.shape[2]):
                        masks_[masks[:,:,j] > 0] = 255
                        
                # Load bboxes from instance when processing blur images
                if BBOX_FLAG and NOISY_FLAG:
                    bbox.generate_bbox_data(output_path, data_ids, OBJ_FLAG, bboxes)
                # Load bboxes from bbox files when processing original images
                elif BBOX_FLAG:
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
                    
                
                
                # Save RGB images
                if OBJ_FLAG:
                    folder = 'object'
                    data_id = data_ids['obj_id']
                else:
                    folder = 'non_object'
                    data_id = data_ids['non_obj_id']
                
                # write the blur images
                rgb_fn_new = os.path.join(output_path, folder, 'images', f"{data_id}.jpg")
                
                if NOISY_FLAG:
                    cv2.imwrite(rgb_fn_new, rgb_blur)
                else:
                    cv2.imwrite(rgb_fn_new, rgb_resized)
                
                # Record the mapping relations
                f3.write('%s  %s\n' %(os.path.join(d, f'{i}.png'), os.path.join(folder, f"{data_id}.png")))
                print(os.path.join(d, f'{i}.png'), " -> ", os.path.join(folder, f"{data_id}.png"))
                
                #del instances, bboxes, masks
                #del rgb, depth, rgb_resized
                
                # visualize the image result
                rgb_ = rgb_blur.copy()
                rgb__ = visualize(rgb_, masks_)
                rgb_ = bbox.colorize_bboxes(bboxes, rgb_)
                cv2.imshow('bbox', rgb_)
                cv2.imshow('mask', rgb__)
                cv2.waitKey(1)
            
            # Close the files for mapping relations and ignored ids
            f2.close()
            f3.close()
            
            # Save the semantic mask json data
            anno_path_object = os.path.join(output_path, 'object','masks', f"{d}_annos_gt.json")
            anno_path_non_object = os.path.join(output_path, 'non_object','masks', f"{d}_annos_gt.json")
            json.dump(instance.annotations_obj, open(anno_path_object, "w+"))
            json.dump(instance.annotations_non_obj, open(anno_path_non_object, "w+"))



if __name__ == '__main__':
    # Define parser arguments
    parser = argparse.ArgumentParser(description="Dataset Generation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to Config File")
    args, _ = parser.parse_known_args()
    
    # load configuration file
    config = confuse.Configuration("Data Generation", __name__)
    config.set_file(args.config)
    config.set_args(args)
    
    main(config)