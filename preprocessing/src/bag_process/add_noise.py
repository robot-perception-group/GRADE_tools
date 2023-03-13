import cv2
import os
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from pyquaternion import Quaternion
from cv_bridge import CvBridge

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
        self.blur_enable = self.config['blur']['enable'].get() # generate blur images
        self.blur_save_enable = self.config['blur']['save']['enable'].get() # save blur params

        # Check the direction of bag folder
        self.bag_dir = self.config['path'].get()
        self.bags = os.listdir(os.path.join(self.bag_dir,'reindex_bags'))
        self.bags.sort()
            
        self.noisy_bag_dir = os.path.join(self.bag_dir, 'noisy_bags')
        if not os.path.exists(self.noisy_bag_dir):
            os.makedirs(self.noisy_bag_dir)
            
            
    def initial_topics(self):
        # Define the signal topic
        if self.config['signal_topic'].get() == self.config['topics']['signal_topic'].keys()[0]:
            self.signal_topic = self.config['signal_topic'].get()
            print('Signal Topic: ', self.signal_topic)
        else:
            raise ValueError('The input Signal Topic does not match the config file,,,')
        
        # Define RGB image topic
        self.rgb_topics = self.config['topics']['rgb_topic'].keys()
        print('RGB Image Topics: ', self.rgb_topics)

        # Define depth image topic
        self.depth_topics = self.config['topics']['depth_topic'].keys()
        print('Depth Image Topics: ', self.depth_topics)

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
        self.rgb_ts = {}
        for i in range(len(self.rgb_topics)):
            self.rgb_ts[self.rgb_topics[i]] = []
        
        self.odom_ts = []
        self.pose_ts = []
        
        self.camera_imus = []
        self.camera_poses = []
        self.odom_lin_vels = []
        
        # Blur Class for RGB Images
        self.blur_proc = {}
        for i in range(len(self.rgb_topics)):
            self.blur_proc[self.rgb_topics[i]] = blur_model.Blur(self.config, self.seed)

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
                    
                if topic in self.rgb_topics:
                    t_img = t.to_sec()
                    self.rgb_ts[topic].append(t_img)

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
        for rgb_topic in self.rgb_topics:
            self.blur_proc[rgb_topic].generate_IMU(self.camera_imus, self.imu_cam_ts, self.rgb_ts[rgb_topic])

        # Update the rgb timestamps: deleting the ignored index
        for rgb_topic in self.rgb_topics:
            for i in self.blur_proc[rgb_topic].rgb_ignore:
                print('Image Blur: Ignore RGB_IMAGE at time: %.4f' %(i))
                self.rgb_ts[rgb_topic].remove(i)

        if self.blur_save_enable:
            for idx, rgb_topic in enumerate(self.rgb_topics):
                self.blur_proc[rgb_topic].output_dir = os.path.join(self.noisy_bag_dir, self.config['blur']['save']['output_dirs'][idx].get())
                if not os.path.exists(self.blur_proc[rgb_topic].output_dir):
                    os.makedirs(self.blur_proc[rgb_topic].output_dir)
                    
                    
    def blur_image(self, msg, t_img, blur, rgb_ts):
        # Store the Message Header
        msg_header = msg.header
        
        # Read the RGb images and generate corresponding H matrice
        img = self.bridge.imgmsg_to_cv2(msg,'rgb8')
        
        '''TO DO: Check the calculation of initial velocity'''        
        if t_img not in blur.rgb_ignore:
            # Calculate the initial velocity given specific timestamp
            v_init = blur_model.velocity_transform(t_img, self.odom_ts, self.odom_lin_vels,
                                                    self.pose_ts, self.camera_poses)
            
            index = rgb_ts.index(t_img)

            # Create blur images
            Hs = blur.blur_homography(index, v_init)
            blur_img = blur.create_blur_image(img, Hs)
            
            # output blur params
            if self.blur_save_enable:
                blur.save_params(index)
        else:
            blur_img = img
        
        # Define the new image msg
        msg = self.bridge.cv2_to_imgmsg(blur_img, 'rgb8')
        msg.header = msg_header
        
        return msg
    
    
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

    # @staticmethod
    # def tf_transform(msg):
    #     transforms_new = []
    #     for m in msg.transforms:
    #         if 'fake/' in m.header.frame_id:
    #             continue
            
    #         # if m.header.frame_id == 'my_robot_0/pitch_link':
    #         #     m.child_frame_id = m.child_frame_id + '_gt'
            
    #         transforms_new.append(m)   
    #     msg.transforms = transforms_new

    #     return msg
    
    
    def play_bags(self):
        if self.blur_enable == True:
            self.blur_preprocess()
            
        for bag in self.bags:
            if '.bag' not in bag or 'orig' in bag:
                continue
            
            print('Adding Noise to the bag', bag)
            
            bag_path = os.path.join(self.bag_dir,'reindex_bags', bag)
            bag = rosbag.Bag(bag_path)
            
            # Noisy rosbags
            w_bag = rosbag.Bag(os.path.join(self.noisy_bag_dir,
                                            f"{bag.filename.split('/')[-1][:-4]}_noisy.bag"), "w")
                    
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
                    if topic in self.rgb_topics:                        
                        # Some RGB Image does not contain enough IMU data
                        t_img = t.to_sec()
                        msg = self.blur_image(msg, t_img, self.blur_proc[topic], self.rgb_ts[topic])
                        

                '''Add noise to IMU'''
                if 'imu' in topic:
                    topic_type = 'imu'
                    
                    if topic not in self.noise_models:
                        # todo check noise model is valid and correctly init
                        self.noise_models[topic] = sensor_model.SensorModel(topic_type, self.config[topic_type]['noise_model'].get(),
                                                                    self.config[topic_type]['config'].get(), self.seed)
                    
                    data = sensor_model.IMU(msg.angular_velocity, msg.linear_acceleration, t)
                    
                    msg.angular_velocity, msg.linear_acceleration = self.noise_models[topic].callback(data)
                
                
                '''Add noise to Depth Image'''
                if 'depth' in topic and topic in self.depth_topics:
                    # Extract message header
                    header = msg.header
                    topic_type = 'camera'
                    
                    if topic not in self.noise_models:
                        # todo check noise model is valid and correctly init
                        self.noise_models[topic] = sensor_model.SensorModel(topic_type, self.config[topic_type]['noise_model'].get(),
                                                                    self.config[topic_type]['config'].get(), self.seed)
                        
                    data = sensor_model.Image(self.bridge.imgmsg_to_cv2(msg, 'passthrough'),  pointcloud_enable=self.pointcloud_enable)
                    x, y, z, depth = self.noise_models[topic].callback(data)
                    # todo finish up kinect case with correct image
                    
                    '''Debug for visualization of PCD'''
                    # if '/1/' in topic and t.to_sec()>5.0:
                    #     cv2.imshow('depth', depth)
                    #     cv2.waitKey(10)
                    #     fig = plt.figure(0)
                    #     ax = plt_pt.axes(projection='3d')
                    #     ax.scatter(x, y, z, s=0.05)
                    #     plt.show()
                    
                    msg = self.bridge.cv2_to_imgmsg(depth, 'passthrough')
                    msg.header = header
                    
                    '''Output Pointcloud Data'''
                    if self.pointcloud_enable == True:
                        msg_pcd, msg_pcd2 = self.create_pointcloud(x, y, z, header)
                        
                        w_bag.write(topic[:-9]+'pointcloud', msg_pcd, t)
                        w_bag.write(topic[:-9]+'pointcloud2', msg_pcd2, t)
                
                
                # '''Filter the TF Transformation'''           
                # if topic == '/tf':
                #     msg = self.tf_transform(msg)
                
                '''Rewrite the Topic Name'''
                if topic in self.config['mapping'].get():
                    topic_name = self.config['mapping'].get()[topic]
                else:
                    topic_name = topic
                
                # Write the msg into the target topics
                w_bag.write(topic_name, msg, t)
            
            w_bag.close()