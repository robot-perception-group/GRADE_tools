import cv2
import os
import numpy as np
from PIL import Image

from model import sensor_model, blur_model
import shutil

class linear_acceleration:
    def __init__(self, x ,y, z):
        self.x = x
        self.y = y
        self.z = z

class angular_velocity:
    def __init__(self, x ,y, z):
        self.x = x
        self.y = y
        self.z = z

class AddNoise:
    def __init__(self, config):
        self.config = config
        self.path = self.config['path'].get()
        
        if self.config["seed"].get():
            self.seed = config["seed"].get()
        else:
            self.seed = None


        # Define Output Directory
        self.out_dir = self.config['out_dir'].get()
        if self.out_dir == '':
            self.out_dir = self.path

        # Flag
        self.DEPTH_IMG_FLAG = self.config["camera"]["output_depth_img"].get() # if output the full-size images
        self.DEPTH_NOISE_FALG = self.config["camera"]["enable"].get()
        self.BLUR_FLAG = self.config["blur"]["enable"].get()
        self.IMU_NOISE_FLAG = self.config["imu"]["enable"].get()
        self.REPLICATE_FLAG = self.config["replicate"].get()
        
        if self.DEPTH_IMG_FLAG:
            self.MIN_DEPTH = self.config['camera']['config']['minimum_depth'].get()
            self.MAX_DEPTH = self.config['camera']['config']['maximum_depth'].get()
            self.DEPTH_FACTOR = self.config['camera']['config']['depth_factor'].get()
            
            
    def init_dir(self):
        # Initialize Data Directory from RAW DATA DIRECOTRY in DATA EXTRACTION
        self.imu_dirs = []
        for d in self.config["imu"]["folder"].get():
            self.imu_dirs.append(os.path.join(self.config['raw_data_dir'].get(),d))
            # Copy IMU data
            if self.REPLICATE_FLAG:
                print("----------  Replicating IMU Data  ----------")
                dst = os.path.join(self.out_dir, "data", d)
                if not os.path.exists(dst):
                    os.makedirs(dst)
                for f in os.listdir(self.imu_dirs[-1]):
                    shutil.copy(os.path.join(self.imu_dirs[-1],f), dst)

        # Define the imu directory used for image blur
        self.imu_dir_for_blurry = os.path.join(self.config['raw_data_dir'].get(),self.config["blur"]["imu_dir_for_blurry"].get())
        
        # Define the odom directory
        self.odom_dir = os.path.join(self.config['raw_data_dir'].get(),self.config["blur"]["odom_dir_for_blurry"].get())

        # Input Data Dir in VIEWPORT FOLDER
        self.depth_dir = os.path.join(self.path,"depthLinear/")
        self.rgb_dir   = os.path.join(self.path,"rgb/")
        self.pose_dir  = os.path.join(self.path,"camera/")

        if self.REPLICATE_FLAG:
            print("----------  Replicating RGB Data  ----------")
            dst = os.path.join(self.out_dir, "data", "rgb")
            if not os.path.exists(dst):
                os.makedirs(dst)
            for f in os.listdir(self.rgb_dir):
                shutil.copy(os.path.join(self.rgb_dir,f), dst)

            print("----------  Replicating Depth Data  ----------")
            dst = os.path.join(self.out_dir, "data", "depth")
            if not os.path.exists(dst):
                os.makedirs(dst)
            for f in os.listdir(self.depth_dir):
                shutil.copy(os.path.join(self.depth_dir,f), dst)

        # Optional Output Data Dir
        if self.DEPTH_NOISE_FALG:
            # Generate the noisy depth directory
            self.depth_noisy_dir = os.path.join(self.out_dir, "data_noisy","depth")
            print('Depth Noisy Files Directory: ', self.depth_noisy_dir)
            if not os.path.exists(self.depth_noisy_dir):
                os.makedirs(self.depth_noisy_dir)
                
        if self.DEPTH_IMG_FLAG:
            # Generate the depth image directory
            self.depth_img_dir = os.path.join(self.out_dir, "data", "depth")
            print('Depth Images Directory: ', self.depth_img_dir)
            if not os.path.exists(self.depth_img_dir):
                os.makedirs(self.depth_img_dir)
            
            # Generate the noisy depth image directory
            if self.DEPTH_NOISE_FALG:
                self.depth_noisy_img_dir = os.path.join(self.out_dir, "data_noisy", "depth")
                print('Depth Noisy Images Directory: ', self.depth_noisy_img_dir)
                if not os.path.exists(self.depth_noisy_img_dir):
                    os.makedirs(self.depth_noisy_img_dir)
        
        if self.BLUR_FLAG:
            self.rgb_blurry_dir = os.path.join(self.out_dir, "data_noisy", "rgb")
            print('RGB Blurry Images Directory: ',  self.rgb_blurry_dir)
            if not os.path.exists(self.rgb_blurry_dir):
                os.makedirs(self.rgb_blurry_dir)
        
    
    def play_files(self):
        if os.path.exists(self.path):
            self.init_dir()
            
            print("\n ===============  PROCESSING VIEWPORT  ===============\n")
            if self.DEPTH_NOISE_FALG:
                self.add_depth_noise()
            
            if self.DEPTH_IMG_FLAG:
                print("\n ===============  Generating non-noisy Depth Images  ===============")
                # Sorted the depth images in order
                depth_files = os.listdir(self.depth_dir)
                depth_files.sort(key=lambda x:int(x[:-4]))
                for file in depth_files:
                    print("------ Processing %s ------" %file,end='\r')
                    depth = np.load(os.path.join(self.depth_dir,file), allow_pickle=True)
                    
                    fn = os.path.join(self.depth_img_dir, str(int(file[:-4])) + '.png')
                    self.save_image(fn, depth)
            if self.BLUR_FLAG:
                self.add_blur()
        else:
            raise ValueError('Folder does not exist')
        
        if self.IMU_NOISE_FLAG:
            for d in self.imu_dirs:
                self.add_imu_noise(d)

        
    def save_image(self, file_name, depth):       
        mask = np.logical_or((depth < self.MIN_DEPTH),(depth > self.MAX_DEPTH))
        depth[mask] = 0
        
        depth = depth * self.DEPTH_FACTOR
        depth = depth.astype('uint16')
        Image.fromarray(depth).save(file_name)


    def add_depth_noise(self):
        print("\n ===============  Adding noise to Depth File ===============") 
        topic_type = "camera"
        camera_config = self.config[topic_type]["config"].get()
        noise_type = self.config[topic_type]["noise_model"].get()
        
        noise_model = sensor_model.SensorModel(topic_type, noise_type, camera_config, self.seed)

        # Sorted the depth images in order
        depth_files = os.listdir(self.depth_dir)
        depth_files.sort(key=lambda x:int(x[:-4]))
        for file in depth_files:
            print("------ Processing %s ------" %file,end='\r')
            depth = np.load(os.path.join(self.depth_dir, file), allow_pickle=True)
            
            data = sensor_model.Image(depth)
            _, _, _, depth = noise_model.callback(data)
            # todo finish up kinect case with correct image

            # output npy files
            np.save(os.path.join(self.depth_noisy_dir,file), depth)
            
            if self.DEPTH_IMG_FLAG:
                fn = os.path.join(self.depth_noisy_img_dir,str(int(file[:-4])) + '.png')
                self.save_image(fn, depth)

    def add_imu_noise(self, imu_dir):   
        print("\n ==========  Adding noise to IMU  ==========")
        
        imu_noisy_dir = os.path.join(self.out_dir, "data_noisy", os.path.basename(imu_dir))
        print('IMU Camera Noisy Directory: ', imu_noisy_dir)

        if not os.path.exists(imu_noisy_dir):
            os.makedirs(imu_noisy_dir)
            
        imu_files = os.listdir(imu_dir)
        imu_files.sort(key=lambda x:(int(x.split('-')[0][4:]),int(x.split('-')[1][4:-4])))
        
        topic_type = "imu"
        noise_model = sensor_model.SensorModel(topic_type, self.config[topic_type]["noise_model"].get(),
                                                                    self.config[topic_type]["config"].get(), self.seed)
        for imu_file in imu_files:
            # Load IMU Data
            print("------ Processing %s ------" %imu_file,end='\r')
            with open(os.path.join(imu_dir, imu_file),'rb') as f:
                imu = np.load(f, allow_pickle=True)
                
            # Define the angular velocity class and linear acceleration class same as the msg type
            ang_vel = angular_velocity(imu.item()["ang_vel"][0],imu.item()["ang_vel"][1],imu.item()["ang_vel"][2])
            lin_acc = linear_acceleration(imu.item()["lin_acc"][0],imu.item()["lin_acc"][1],imu.item()["lin_acc"][2])
            
            # Add noise to the IMU data
            data = sensor_model.IMU(ang_vel, lin_acc, imu.item()["time"])
            ang_vel_data, accel_data = noise_model.callback(data)
            
            # Save the noisy data
            imu_noisy = {}
            imu_noisy["time"] = imu.item()["time"]
            imu_noisy["ang_vel"] = [ang_vel_data.x, ang_vel_data.y, ang_vel_data.z]
            imu_noisy["lin_acc"] = [accel_data.x, accel_data.y, accel_data.z]
            
            np.save(os.path.join(imu_noisy_dir, imu_file), imu_noisy)

    def add_blur(self):
        print("\n ===============  Generating Blurry RGB Images  ===============")
        
        # load all required data
        _, rgb_ts, imu_ts, imus, odom_ts, odom_lin_vels, poses = self.load_data()
        pose_ts = rgb_ts
        
        # define the blur class
        blur = blur_model.Blur(self.config, self.seed)
        blur.generate_IMU(imus, imu_ts, rgb_ts)
        # generate the blurry image
        for i in range(len(rgb_ts)):
            print("------ Processing %s ------" %(str(i+1)+".png"),end='\r')
            v_init = blur_model.velocity_transform(rgb_ts[i], odom_ts, odom_lin_vels, pose_ts, poses)
            
            img = cv2.imread(self.rgb_dir + str(i+1) + ".png")
            Hs = blur.blur_homography(i, v_init)

            blur_img = blur.create_blur_image(img, Hs)
            cv2.imwrite(os.path.join(self.rgb_blurry_dir,str(i+1)+".png"), blur_img)

    def load_data(self):
        '''Load Data for Blurry Images'''
        imu_files = os.listdir(self.imu_dir_for_blurry)
        imu_files.sort(key=lambda x:(int(x.split('-')[0][4:]),int(x.split('-')[1][4:-4])))
        
        # Define RGB Image list, IMU Timestamps, IMU Data
        rgb_list = []
        imu_ts = []
        imus = []
        for imu_file in imu_files:
            if imu_file.split('-')[0] not in rgb_list:
                rgb_list.append(imu_file.split('-')[0])
                
            with open(os.path.join(self.imu_dir_for_blurry,imu_file),'rb') as f:
                data = np.load(f, allow_pickle=True)
                ang_vel = data.item()["ang_vel"]
                lin_acc = data.item()["lin_acc"]
                imu_ts.append(data.item()["time"])
                imus.append([ang_vel[0],ang_vel[1],ang_vel[2],lin_acc[0],lin_acc[1],lin_acc[2]])
        
        '''Define RGB Image Timestamps based on the IMU Data'''
        rgb_ts = []
        for rgb in rgb_list:
            for imu_file in imu_files:
                if rgb + '-' in imu_file:
                    imu_file_ = imu_file # loop to find the last imu data for this rgb image
            
            # Load the timestamp of the final imu data
            with open(os.path.join(self.imu_dir_for_blurry, imu_file_),'rb') as f:
                data = np.load(f, allow_pickle=True)
                imu_t = data.item()["time"]
                rgb_ts.append(imu_t)
        print("Number of RGB Image from IMU data: ", len(rgb_ts))
        
        '''Load Odom Data'''
        odom_ts = []
        odom_lin_vels = []
        
        odom_files = os.listdir(self.odom_dir)
        odom_files.sort(key=lambda x:int(x[:-4]))
        for odom_file in odom_files:
            with open(os.path.join(self.odom_dir, odom_file), 'rb') as f:
                data = np.load(f, allow_pickle=True)
                odom_ts.append(data.item()["time"])
                odom_lin_vels.append(data.item()["lin_vel"])

        if self.REPLICATE_FLAG:
            print("----------  Replicating Odom Data  ----------")
            dst = os.path.join(self.out_dir, "data", os.path.basename(self.odom_dir))
            if not os.path.exists(dst):
                os.makedirs(dst)
            for f in os.listdir(self.odom_dir):
                shutil.copy(os.path.join(self.odom_dir,f), dst)

        '''Load Pose Data'''
        pose_files = os.listdir(self.pose_dir)
        pose_files.sort(key=lambda x:int(x[:-4]))
        
        poses = []
        for pose_file in pose_files:
            with open(os.path.join(self.pose_dir, pose_file),'rb') as f:
                data = np.load(f, allow_pickle=True)
                pose_camera = data.item()["pose"]
                pose_camera = pose_camera.T
                poses.append(pose_camera[:3,:3])
                
        
        return rgb_list, rgb_ts, imu_ts, imus, odom_ts, odom_lin_vels, poses