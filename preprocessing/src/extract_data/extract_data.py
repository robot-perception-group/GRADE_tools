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
    # IMU index will correlated to the Image Index
    imu_reference_topic = config["imu_reference_topic"].get()

    dirs = []
    rgb_topics = []
    imu_topics = []
    depth_topics = []
    odom_topics = []
    pose_topics = []
    save_rgb_topics = []
    save_imu_topics = []
    save_depth_topics = []
    save_odom_topics = []
    save_pose_topics = []

    '''Define Directories'''
    bag_dir = config['path'].get()

    # define output directory
    out_dir = config['out_dir'].get()
    if out_dir=="":
        out_dir = bag_dir

    rgb_map = {}
    rgb_map_tmp = config['save_images']['rgb_topics'].get()
    depth_map = {}
    depth_map_tmp = config['save_images']['depth_topics'].get()
    for k,v,s in rgb_map_tmp:
        rgb_map[k] = v
        rgb_topics.append(k)
        if s:
            save_rgb_topics.append(k)
            dirs.append(os.path.join(out_dir,f'data/{v}'))
            print('Topic [%s] will be saved in: [%s]' %(k, os.path.join(out_dir,f'data/{v}')))
    for k,v,s in depth_map_tmp:
        depth_map[k] = v
        depth_topics.append(k)
        if s:
            save_depth_topics.append(k)
            dirs.append(os.path.join(out_dir,f'data/{v}'))
            print('Topic [%s] will be saved in: [%s]' %(k, os.path.join(out_dir,f'data/{v}')))

    imu_map = {}
    imu_map_tmp = config['save_files']['imu_topics'].get()
    odom_map = {}
    odom_map_tmp = config['save_files']['odom_topics'].get()
    pose_map = {}
    pose_map_tmp = config['save_files']['pose_topics'].get()
    
    for k,v,s in imu_map_tmp:
        imu_map[k] = v
        imu_topics.append(k)
        if s:
            save_imu_topics.append(k)
            dirs.append(os.path.join(out_dir,f'data/{v}'))
            print('Topic [%s] will be saved in: [%s]' %(k, os.path.join(out_dir,f'data/{v}')))
    for k,v,s in odom_map_tmp:
        odom_map[k] = v
        odom_topics.append(k)
        if s:
            save_odom_topics.append(k)
            dirs.append(os.path.join(out_dir,f'data/{v}'))
            print('Topic [%s] will be saved in: [%s]' %(k, os.path.join(out_dir,f'data/{v}')))
    for k,v,s in pose_map_tmp:
        pose_map[k] = v
        pose_topics.append(k)
        if s:
            save_pose_topics.append(k)
            dirs.append(os.path.join(out_dir,f'data/{v}'))
            print('Topic [%s] will be saved in: [%s]' %(k, os.path.join(out_dir,f'data/{v}')))

    # Check the existence of rosbags
    if not os.path.exists(bag_dir):
        raise ValueError('Please refer to a correct rosbag folder')
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    # Define processed rosbag
    bags = os.listdir(bag_dir)
    bags.sort()

    '''Define Params'''
    # initalize parameters
    rgb_counter = [1] * len(save_rgb_topics)
    depth_counter = [1] * len(save_depth_topics)

    DEPTH_FACTOR = config['save_images']['depth_factor'].get()

    imu_ref_t = []
    
    print("\n ==========  STAGE 1: Load all RGB Timestamps / Save Images  ==========")
    for bag in bags:
        if not bag.endswith(".bag"):
            continue
        
        print("Playing bag", bag)
        
        bag_path = os.path.join(bag_dir, bag)
        bag = rosbag.Bag(bag_path)
        for topic, msg, t in bag.read_messages(topics = rgb_topics + depth_topics + [imu_reference_topic]):
            # Load rgb image timestamps
            if topic == imu_reference_topic:
                imu_ref_t.append(t.to_sec())

            #  Save images from rosbag
            if topic in save_rgb_topics:
                img = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                
                # Save RGB Images
                d = os.path.join(out_dir,"data",rgb_map[topic])
                index = rgb_counter[save_rgb_topics.index(topic)]
                fn = os.path.join(d,str(index) + '.png') # KITTI name format
                Image.fromarray(img).save(fn)
                
                rgb_counter[save_rgb_topics.index(topic)] += 1
            elif topic in save_depth_topics:
                depth = cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
                depth = depth * DEPTH_FACTOR
                depth = depth.astype('uint16')
                
                # Save Depth Images
                index = depth_counter[save_depth_topics.index(topic)]
                d = os.path.join(out_dir,"data",depth_map[topic])
                fn = os.path.join(d,str(index) + '.png')
                Image.fromarray(depth).save(fn)
                
                depth_counter[save_depth_topics.index(topic)] += 1
        bag.close()
        
    print("\n ==========  Stage 2: Save the IMU / Odom for RGB Images  ==========")
    
    imu_img_idx = [1] * len(save_imu_topics)
    odom_counter = [1] * len(save_odom_topics)
    pose_counter = [1] * len(save_pose_topics)
    imu_counter = [1] * len(save_imu_topics)

    for bag in bags:
        if not bag.endswith(".bag"):
            continue

        print("Playing bag", bag)
        
        bag_path = os.path.join(bag_dir, bag)
        bag = rosbag.Bag(bag_path)
        new_idx = -1
        for topic, msg, t in bag.read_messages(topics = save_imu_topics + [imu_reference_topic] + save_odom_topics + save_pose_topics):
            if topic in save_imu_topics:
                imu_t = t.to_sec()

                for i in range(0, len(imu_ref_t)):
                    if imu_t <= imu_ref_t[i] + 10**(-5):
                        new_idx = "img_"+str(i+1)
                        break

                # Update the img index
                if imu_img_idx[save_imu_topics.index(topic)] != new_idx:
                    imu_img_idx[save_imu_topics.index(topic)] = new_idx
                    imu_counter[save_imu_topics.index(topic)] = 1
                else:
                    imu_counter[save_imu_topics.index(topic)] += 1

                # Update the imu index
                imu_index = "imu_"+str(imu_counter[save_imu_topics.index(topic)])

                # Define File Name
                fn= new_idx + "-" + imu_index + ".npy"

                # Load Data
                imu = {}
                imu["time"] = imu_t
                imu["ang_vel"] = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
                imu["lin_acc"] = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]

                # Save Message
                d = os.path.join(out_dir,"data",imu_map[topic]) # directory
                np.save(os.path.join(d, fn), imu)
                
            elif topic in save_odom_topics:
                odom_t = t.to_sec()

                # Define File Name
                fn = str(odom_counter[save_odom_topics.index(topic)]) + ".npy"
                odom_counter[save_odom_topics.index(topic)] += 1

                odom = {}
                odom["time"] = odom_t
                odom["position"] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
                odom["orientation"] = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
                odom["lin_vel"] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
                odom["ang_vel"] = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]

                # Save Message
                d = os.path.join(out_dir,"data", odom_map[topic]) # directory
                np.save(os.path.join(d,fn), odom)
            elif topic in save_pose_topics:
                pose_t = t.to_sec()

                # Define File Name
                fn = str(pose_counter[save_pose_topics.index(topic)]) + ".npy"
                pose_counter[save_pose_topics.index(topic)] += 1

                pose = {}
                pose["time"] = pose_t
                pose["position"] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
                pose["orientation"] = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

                # Save Message
                d = os.path.join(out_dir,"data", pose_map[topic]) # directory
                np.save(os.path.join(d,fn), pose)
        bag.close()