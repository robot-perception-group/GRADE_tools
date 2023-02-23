import cv2
import os
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from pyquaternion import Quaternion
from cv_bridge import CvBridge
from PIL import Image

from model import sensor_model, blur_model

class AddNoise:
    def __init__(self, config):
        print('\n\n ==============  INITIALIZATION  ==============')
        # Define Configuration Values
        self.config = config
        
        # Initialize the parameters
        self.bridge = CvBridge()
        self.noise_models = {}
        
        # Define required input topics
        self.initial_topics()
        
        # Define the random seed
        if self.config['seed'].get():
            self.seed = self.config['seed'].get()
        else:
            self.seed = None
        
        # Define Enable FLAGs
        self.started = False
        self.noise_enable = self.config['noise'].get()
        self.pointcloud_enable = self.config['camera']['pointcloud'].get()
        self.blur_enable = self.config['blur']['enable'].get()

        # Check the direction of bag folder
        self.bag_dir = self.config['path'].get()
        self.bags = os.listdir(os.path.join(self.bag_dir,'reindex_bags'))
        self.bags.sort()
            
        self.noisy_dir = os.path.join(self.bag_dir, 'Viewport0_occluded/rgb')
        if not os.path.exists(self.noisy_dir):
            os.makedirs(self.noisy_dir)
            
        self.blur_dir = os.path.join(self.bag_dir, 'Viewport0_occluded/blur')
        if not os.path.exists(self.blur_dir):
            os.makedirs(self.blur_dir)
            
        self.idx = 1
            
            
    def initial_topics(self):
        # Define the signal topic
        if self.config['signal_topic'].get() == self.config['topics']['signal_topic'].keys()[0]:
            self.signal_topic = self.config['signal_topic'].get()
            print('Signal Topic: ', self.signal_topic)
        else:
            raise ValueError('The input Signal Topic does not match the config file,,,')
        
        # Define RGB image topic
        self.rgb_topic_0 = self.config['topics']['rgb_topic'].keys()[0]
        self.rgb_topic_1 = self.config['topics']['rgb_topic'].keys()[1]
        print('RGB Image Topic: ', self.rgb_topic_0)
        print('RGB Image Topic: ', self.rgb_topic_1)
        
        # Define depth image topic
        self.depth_topic_0 = self.config['topics']['depth_topic'].keys()[0]
        self.depth_topic_1 = self.config['topics']['depth_topic'].keys()[1]
        print('Depth Image Topic: ', self.depth_topic_0)
        print('Depth Image Topic: ', self.depth_topic_1)
        
        self.odom_topic = self.config['topics']['odom_topic'].keys()[0]
        self.imu_camera_topic = self.config['topics']['imu_camera_topic'].keys()[0]
        self.pose_camera_topic = self.config['topics']['pose_camera_topic'].keys()[0]
        self.tf_topic = self.config['topics']['tf_topic'].keys()[0]
        print('Odom Topic:', self.odom_topic)
        print('IMU Camera Topic:', self.imu_camera_topic)
        print('Pose Camera Topic:', self.pose_camera_topic)
        print('TF Topic:', self.tf_topic)
        
        # Define other topics need to be stored
        self.topics = []
        print('\nTopics Input: ')
        for topic_type in self.config['topics'].keys():
            for topic_name in self.config['topics'][topic_type].keys():
                self.topics.append(topic_name)
                print(topic_name)
                
                
    def blur_preprocess(self):
        # Load Blur Parameters 
        self.imu_cam_ts = []
        self.rgb_0_ts = []
        self.rgb_1_ts = []
        self.odom_ts = []
        self.pose_ts = []
        
        self.camera_imus = []
        self.camera_poses = []
        self.odom_lin_vels = []
        
        # Blur Class for RGB Images
        self.blur_0 = blur_model.Blur(self.config, self.seed)
        self.blur_1 = blur_model.Blur(self.config, self.seed)
        
        '''First Loop: Generate new Timestamp and Interpolation for IMU data''' 
        for bag in self.bags:
            if '.bag' not in bag or 'orig' in bag:
                continue
            
            print('Preprocssing ',bag,'for Motion Blur')
            
            bag_path = os.path.join(self.bag_dir,'reindex_bags', bag)
            bag = rosbag.Bag(bag_path)
            
            for topic, msg, t in bag.read_messages(topics=self.topics):              
                if topic == self.imu_camera_topic :
                    # Define the timestamp for all imu_camera data
                    t_imu = t.to_sec()
                    self.imu_cam_ts.append(t_imu)
                    
                    # Define the imu camera data
                    omega = msg.angular_velocity
                    acc = msg.linear_acceleration
                    self.camera_imus.append([omega.x, omega.y, omega.z, acc.x, acc.y, acc.z])
                    
                if topic == self.rgb_topic_0:
                    t_img = t.to_sec()
                    self.rgb_0_ts.append(t_img)
                
                if topic == self.rgb_topic_1:
                    t_img = t.to_sec()
                    self.rgb_1_ts.append(t_img)
                    
                '''Calculate the Initial Velocity for Each Frame'''
                if topic == self.odom_topic:
                    t_odom = t.to_sec()
                    self.odom_ts.append(t_odom)
                    
                    self.odom_lin_vels.append([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
                    
                if topic == self.pose_camera_topic:
                    t_pose_cam = t.to_sec()
                    self.pose_ts.append(t_pose_cam)
                    
                    rot = Quaternion([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]).rotation_matrix
                    self.camera_poses.append(rot)
        
        '''Interpolte IMU data for each RGB Frame'''
        #self.blur_0.generate_IMU(self.camera_imus, self.imu_cam_ts, self.rgb_0_ts)
        self.blur_1.generate_IMU(self.camera_imus, self.imu_cam_ts, self.rgb_1_ts)
        
        for i in self.blur_1.rgb_ignore:
            print('Image Blur: Ignore RGB_IMAGE_1 at time: %.4f' %(i))
            self.rgb_1_ts.remove(i)
        
        
    def blur_image(self, msg, t_img, blur, rgb_ts):
        # Store the Message Header
        msg_header = msg.header
        
        # Read the RGb images and generate corresponding H matrice
        img0 = self.bridge.imgmsg_to_cv2(msg,'rgb8')
        
        '''TO DO: Check the calculation of initial velocity'''        
        if t_img not in blur.rgb_ignore:
            # Calculate the initial velocity given specific timestamp
            v_init = blur_model.velocity_transform(t_img, self.odom_ts, self.odom_lin_vels,
                                                    self.pose_ts, self.camera_poses)
            
            index = rgb_ts.index(t_img)
        
            Hs, H_mean = blur.blur_homography(index, v_init)
            # Create blur images
            img0 = blur.create_blur_image(img0, Hs)
        
        rgb_resized = cv2.resize(img0, dsize=(960, 720))
        
        idx = self.idx
        fn = os.path.join(self.noisy_dir,f"{idx}.jpg")
        
        blur_data = {}
        blur_data['exposure_time'] = blur.exposure_time
        blur_data['readout_time'] = blur.t_readout
        blur_data['interval'] = blur.interval
        blur_data['num_pose'] = blur.num_pose
        blur_data['H_mean'] = H_mean
        blur_data['extrinsic_mats'] = blur.extrinsic_mats
        blur_data['intrinsic_mat'] = blur.intrinsic_mat
        
        # Save Output Files
        np.save(os.path.join(self.blur_dir,f"{idx}.npy"), blur_data)
        Image.fromarray(rgb_resized).save(fn)
        
        self.idx += 1
    
    def create_pointcloud(self, x, y, z, header):
        points = np.stack([x, y, z], axis=-1)
                            
        '''PointCloud()'''
        msg_pcd = PointCloud()
        msg_pcd.header = header
                                
        for i in range(x.shape[0]):
            msg_pcd.points.append(Point32(x[i],y[i],z[i]))
                                
        '''PointCloud2()'''
        msg_pcd2 = pcl2.create_cloud_xyz32(header, points)
        
        return msg_pcd, msg_pcd2
    
    
    def play_bags(self):
        if self.blur_enable == True:
            self.blur_preprocess()
            
        for bag in self.bags:
            if '.bag' not in bag or 'orig' in bag:
                continue
            
            print('Adding Noise to the bag', bag)
            
            bag_path = os.path.join(self.bag_dir,'reindex_bags', bag)
            bag = rosbag.Bag(bag_path)
                    
            '''Create New ROS bags'''
            for topic, msg, t in bag.read_messages(topics=self.topics):
                
                # Waiting for the start signal
                if topic == self.signal_topic:
                    if msg.data == 'starting':
                        self.started = True
                        print('Starting...')
                    elif msg.data == 'stop':
                        self.started = False
                        
                if not self.started:
                    continue
                    
                '''Add blur to RGB images'''
                if self.blur_enable == True:
                    if topic == self.rgb_topic_1:                         
                        # Some RGB Image does not contain enough IMU data
                        t_img = t.to_sec()
                        
                        # Generate blur images and save it as JPG files                        
                        self.blur_image(msg, t_img, self.blur_1, self.rgb_1_ts)
        
            