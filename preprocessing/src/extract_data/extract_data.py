import os
import cv2
import rosbag
import numpy as np
from cv_bridge import CvBridge
from PIL import Image


def extract_data(config):
    # Define CV bridge for image transfer
    cv_bridge = CvBridge()
    
    '''Define Required Topic Name'''
    topics = config['topics'].get()
    
    SAVE_IMAGE = False
    SAVE_FILE = False
    if config['save_images']['enable'].get():
        SAVE_IMAGE = True 
        save_image_topics = config['extract_bag']['save_images']['topics'].get()
        
    if config['save_files']['enable'].get():
        SAVE_FILE = True 
        save_file_topics = config['extract_bag']['save_files']['topics'].get()
    
    '''Define Directories'''
    #'reindex' for outputting original data, 'noisy' for outputting noisy data
    bag_dir = config['path'].get()
    if 'noisy' in bag_dir:
        dir_suffix = '_noisy/'
    else:
        dir_suffix = '/'
    
    if SAVE_FILE:
        imu_camera_dir = bag_dir + '/../data/imu_camera' + dir_suffix
        imu_body_dir = bag_dir  + '/../data/imu_body' + dir_suffix
        odom_dir = bag_dir  + '/../data/odom' + dir_suffix
        
    if SAVE_IMAGE:
        rgb_1_dir = bag_dir  + '/../data/rgb' + dir_suffix # output resized rgb image (640X480)
        depth_1_dir = bag_dir  + '/../data/depth' + dir_suffix # output resized depth image (640X480)
        rgb_2_dir = bag_dir  + '/../data/rgb_occluded' + dir_suffix # output resized rgb image (640X480)
        depth_2_dir = bag_dir  + '/../data/depth_occluded' + dir_suffix # output resized depth image (640X480)
    
    # Check the existence of rosbags
    if not os.path.exists(bag_dir):
        raise ValueError('Please refer to a correct rosbag folder')
    
    # Create the output directories
    dirs = [imu_camera_dir, imu_body_dir, odom_dir, rgb_1_dir, rgb_2_dir, depth_1_dir, depth_2_dir]
    for d in dirs:
        if (d != None) and (not os.path.exists(d)):
            os.makedirs(d)

    # Define processed rosbag
    bags = os.listdir(bag_dir)
    bags.sort()

    '''Define Params'''
    # List of RGB view Timestamp
    rgb1_ts = []
    rgb2_ts = []
    
    # initalize parameters
    index_rgb_1 = 0
    index_rgb_2 = 0
    index_dep_1 = 0
    index_dep_2 = 0
    
    DEPTH_FACTOR = config['save_images']['depth_factor'].get()
    
    
    print("\n ==========  STAGE 1: Load all RGB Timestamps / Save Images  ==========")
    for bag in bags:
        if ".bag" not in bag:
            continue
        
        if ".active" in bag or ".orig" in bag:
            continue
        
        print("Playing bag", bag)
        
        bag_path = os.path.join(bag_dir, bag)
        bag = rosbag.Bag(bag_path)

        for topic, msg, t in bag.read_messages(topics = topics.values()):
            # Load rgb image timestamps
            if topic == topics['rgb_topic_1']:
                rgb1_t = t.to_sec()
                rgb1_ts.append(rgb1_t)  

            elif topic == topics['rgb_topic_2']:
                rgb2_t = t.to_sec()
                rgb2_ts.append(rgb2_t)
            
            #  Save images from rosbag
            if SAVE_IMAGE and topic in save_image_topics:
                if topic == topics['rgb_topic_1']:
                    img = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                    fn = rgb_1_dir + str(index_rgb_1).zfill(6) + '.png' # KITTI name format
                    Image.fromarray(img).save(fn)
                    index_rgb_1 += 1
                    
                elif topic == topics['rgb_topic_2']:
                    img = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                    fn = rgb_2_dir + str(index_rgb_2).zfill(6) + '.png'
                    Image.fromarray(img).save(fn)
                    index_rgb_2 += 1
                    
                elif topic == topics['depth_topic_1']:
                    depth = cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
                    depth = depth * DEPTH_FACTOR
                    depth = depth.astype('uint16')
                    fn = depth_1_dir + str(index_dep_1).zfill(6) + '.png'
                    Image.fromarray(depth).save(fn)
                    index_dep_1 += 1
                    
                elif topic == topics['depth_topic_2']:
                    depth = cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
                    depth = depth * DEPTH_FACTOR
                    depth = depth.astype('uint16')
                    fn = depth_2_dir + str(index_dep_2).zfill(6) + '.png'
                    Image.fromarray(depth).save(fn)
                    index_dep_2 += 1
        bag.close()
        
    print("\n ==========  Stage 2: Save the IMU / Odom for RGB Images  ==========")
    
    camera_index = None
    body_index = None
    odom_index = 1
    
    for bag in bags:
        if ".bag" not in bag:
            continue
        if ".active" in bag or ".orig" in bag:
            continue
        
        print("Playing bag", bag)
        
        bag_path = os.path.join(bag_dir, bag)
        bag = rosbag.Bag(bag_path)

        for topic, msg, t in bag.read_messages(topics = save_file_topics):            
            if SAVE_FILE:
                if topic == topics['imu_camera_topic']:
                    imu_t = t.to_sec()
                    
                    for i in range(len(rgb1_ts)):
                        rgb_t = rgb1_ts[i]
                        if imu_t <= rgb_t:
                            camera_index_new = "img_"+str(i+1)
                            break
                    
                    # Update the img index
                    if camera_index != camera_index_new:
                        camera_index = camera_index_new
                        j = 1
                    else:
                        j += 1
                    
                    # Update the imu index
                    imu_index = "imu_"+str(j)
                    
                    # Define File Name    
                    fn= camera_index + "-" + imu_index + ".npy"
                    
                    # Load Data
                    imu = {}
                    imu["time"] = imu_t
                    imu["ang_vel"] = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
                    imu["lin_acc"] = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
                    
                    # Save Message
                    np.save(imu_camera_dir + fn, imu)
                
                elif topic == topics['imu_body_topic']:
                    imu_t = t.to_sec()
                    
                    for i in range(len(rgb1_ts)):
                        rgb_t = rgb1_ts[i]
                        if imu_t <= rgb_t:
                            body_index_new = "img_"+str(i+1)
                            break
                    
                    # Update the img index
                    if body_index != body_index_new:
                        body_index = body_index_new
                        k = 1
                    else:
                        k += 1
                    
                    # Update the imu index
                    imu_index = "imu_"+str(k)
                    
                    # Define File Name    
                    fn = body_index + "-" + imu_index + ".npy"
                    
                    # Load Data
                    imu = {}
                    imu["time"] = imu_t
                    imu["ang_vel"] = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
                    imu["lin_acc"] = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
                    
                    # Save Message
                    np.save(imu_body_dir + fn, imu)
                    
                elif topic == topics['odom_topic']:
                    odom_t = t.to_sec()
                    
                    # Define File Name    
                    fn = str(odom_index) + ".npy"
                    
                    odom = {}
                    odom["time"] = odom_t
                    odom["position"] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
                    odom["orientation"] = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
                    odom["lin_vel"] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
                    odom["ang_vel"] = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
                    
                    # Save Message
                    np.save(odom_dir + fn, odom)
                    odom_index += 1
        bag.close()