import os
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation

# Transform the velocity from world frame to the camera frame
def velocity_transform(t_img, ts_odom, odom_vels, ts_rot_cam, rots_cam):
    # Find the index of current time in /odom list
    index = np.abs(t_img - np.array(ts_odom)).argmin()
    linear_vel_world = np.array(odom_vels[index])
    
    # Find the index of current time in /pose list
    index = np.abs(t_img - np.array(ts_rot_cam)).argmin()
    trans_mat = np.array(rots_cam[index])
    
    # Obtain the current velocity in camera frame
    linear_vel = np.matmul(np.linalg.inv(trans_mat),linear_vel_world)
    
    return linear_vel
    
    
class Blur(object):
    def __init__(self, params, seed):
        # Define random number
        self.seed = seed
        self.rng = np.random.default_rng(seed = self.seed)
        
        # Parameter Initializaton
        self.blur_params = params["blur"]["config"].get()
        self.camera_params = params["camera"]["config"]["camera_params"].get()
        
        self.imu_sample_freq = params["imu"]["config"]["frequency"].get() # IMU Sampleing Frequency
        self.num_imu_sample = self.blur_params["num_imu_sample"] # Number of IMu data per interval
        
        # If num_pose != num_sample, it will interpolate the imu data
        if isinstance(self.blur_params["exposure_time"],list):
            print("got an interval")
            self.exposure_time = self.rng.uniform(self.blur_params["exposure_time"][0], self.blur_params["exposure_time"][1]) # Time Interval for recording camera motion
        else:
            self.exposure_time = self.blur_params["exposure_time"]
        self.num_pose = self.blur_params["num_pose"] # Number of poses during the exposure time
        self.interval = self.exposure_time / self.num_pose

        self.readout_mean = self.blur_params["readout_mean"]
        self.readout_std = self.blur_params["readout_std"]
        
        # Image Property
        self.image_W = self.camera_params[0]
        self.image_H = self.camera_params[1]
        self.total_sub = self.image_H
        self.intrinsic_mat = np.array([[self.camera_params[2], 0, self.camera_params[4]],
                                       [0, self.camera_params[3], self.camera_params[5]],
                                       [0, 0, 1]])
        
        # Initialize the list of IMU data for each RGB image 
        self.gyro_xs = []
        self.gyro_ys = []
        self.gyro_zs = []
        self.acc_xs = []
        self.acc_ys = []
        self.acc_zs = []
        
        # Initialize the list of RGB image need to be ignored
        self.rgb_ignore = []
        
        # Initialize output directory for blur params
        self.output_dir = None
    
    def generate_IMU(self, imu_camera, imu_camera_timestamps, rgb_timestamps):
        '''
        :param imu_camera: list of all imu camera data
        :param imu_camera_timestamp: list of the timestamp of each imu camera data
        :param rgb_timestamp: list of the timestamp of publishing each rgb image
        
        Select the imu data within the exposure time of each RGB image, and return the interpoleted data.
        '''
        imu_camera = np.array(imu_camera)
        
        for i in range(len(rgb_timestamps)):          
            init_t = rgb_timestamps[i] - self.exposure_time
            end_t = rgb_timestamps[i]
            
            # Define the sample data from imu for each interval
            imu_sample = []
            timestamps_old = []
            
            timestamps = np.array([k * self.interval + init_t for k in range(self.num_pose+1)])
            
            for j in range(len(imu_camera_timestamps)):
                if imu_camera_timestamps[j] >= init_t-10**(-4) and imu_camera_timestamps[j] <= end_t+10**(-4):
                    timestamps_old.append(imu_camera_timestamps[j])
                    imu_sample.append(imu_camera[j])
            
            #print("Length: ",len(timestamps_old))
            if len(timestamps_old) == 0:
                self.rgb_ignore.append(rgb_timestamps[i])
                continue
                
                    
            raw_gyro_x = np.array([imu_sample[k][0] for k in range(len(imu_sample))])
            raw_gyro_y = np.array([imu_sample[k][1] for k in range(len(imu_sample))])
            raw_gyro_z = np.array([imu_sample[k][2] for k in range(len(imu_sample))])
            raw_acc_x = np.array([imu_sample[k][3] for k in range(len(imu_sample))])
            raw_acc_y = np.array([imu_sample[k][4] for k in range(len(imu_sample))])
            raw_acc_z = np.array([imu_sample[k][5] for k in range(len(imu_sample))])
            
            gyro_x = np.interp(timestamps, np.array(timestamps_old), raw_gyro_x)
            gyro_y = np.interp(timestamps, np.array(timestamps_old), raw_gyro_y)
            gyro_z = np.interp(timestamps, np.array(timestamps_old), raw_gyro_z)
            acc_x = np.array([self.nearest_acc(t, raw_acc_x, timestamps_old) for t in timestamps])
            acc_y = np.array([self.nearest_acc(t, raw_acc_y, timestamps_old) for t in timestamps])
            acc_z = np.array([self.nearest_acc(t, raw_acc_z, timestamps_old) for t in timestamps])
            
            self.gyro_xs.append(gyro_x)
            self.gyro_ys.append(gyro_y)
            self.gyro_zs.append(gyro_z)
            self.acc_xs.append(acc_x)
            self.acc_ys.append(acc_y)
            self.acc_zs.append(acc_z)
        
    
    def nearest_acc(self, t, acc, timestamps_old):
        '''
        :param t: single new timestamp for interpolation
        :param acc: orginal IMU linear acceleration data
        :param timestamps_old: original timestamps for IMU data
        
        :return nearest acceleration data
        '''
        i_nearest = np.abs(t - np.array(timestamps_old)).argmin()
        return acc[i_nearest]
        
    def compute_rotations(self, index):
        rotations = []
        dt = self.interval
        R = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
        
        for i in range(self.num_pose + 1):
            omega_xt, omega_yt, omega_zt = self.gyro_xs[index][i], self.gyro_ys[index][i], self.gyro_zs[index][i]
            rotation_operator = np.array([[           1, -omega_zt*dt,  omega_yt*dt],
                                          [ omega_zt*dt,            1, -omega_xt*dt],
                                          [-omega_yt*dt,  omega_xt*dt,            1]])
            
            R = np.matmul(rotation_operator, R)

            rotations.append(R)
        
        # calculate the average rotation
        euler14 = Rotation.from_matrix(rotations[14]).as_euler('zyx',degrees=False)
        euler15 = Rotation.from_matrix(rotations[15]).as_euler('zyx',degrees=False)
        euler_avg = np.arctan((np.sin(euler14)+np.sin(euler15))/(np.cos(euler14)+np.cos(euler15)))

        rot = Rotation.from_euler('zyx', euler_avg, degrees=False)
        rotation_mean = rot.as_matrix()

        return rotations, rotation_mean
    
    def compute_translations(self, rotations, index, v_init):
        translations = []
        dt = self.interval
        T0_star = np.array([0, 0, 0]).reshape(3, 1)
        T = np.array([0, 0, 0]).reshape(3, 1)
        T_star = T0_star
        v = np.array(v_init).reshape(3, 1)
        #print(v)
        translations.append(T)
        
        for i in range(1, self.num_pose+1):
            a_ = np.array([self.acc_xs[index][i-1], self.acc_ys[index][i-1], self.acc_zs[index][i-1]]).reshape(3, 1)
            a  = np.array([self.acc_xs[index][i], self.acc_ys[index][i], self.acc_zs[index][i]]).reshape(3, 1)
            invR_ = np.linalg.inv(rotations[i-1])
            invR  = np.linalg.inv(rotations[i])
            R = rotations[i]
            v_ = v
            
            v = v_ + (np.matmul(invR_, a_) + np.matmul(invR, a)) * dt / 2
            T_star = T_star + (v_ + v) * dt / 2
            T = T_star - np.matmul(R, T0_star)

            translations.append(T)
            
        # calculate the average translation for generating masks for blurred images
        translation_mean = np.mean(translations[14:16],axis=0)

        return translations, translation_mean
        
            
        
    def blur_homography(self, index, v_init):
        rotations,rotation_mean = self.compute_rotations(index)
        translations,translation_mean = self.compute_translations(rotations, index, v_init)
        # print("\n Rotation: \n", self.rotations)
        # print("\n Translation: \n",self.translations)
        
        norm_v = np.array([0, 0, 1]).reshape(1,3)
        
        K = self.intrinsic_mat
        Hs = []
        extrinsic_mats = []
        extrinsic_mats.append(np.array([[1,0,0],[0,1,0],[0,0,1]]))
        
        for i in range(1, self.num_pose+1):
            R = rotations[i]  # np.matmul(rotations[i], Ri3_0)
            T = np.matmul(translations[i], norm_v)
            E = R + T
            E = E / E[2][2]
            H = np.matmul(np.matmul(K, R+T), np.linalg.inv(K))
            H = H / H[2][2]
            extrinsic_mats.append(E)
            Hs.append(H)
        
        # calculate the average matrix for generating masks for blurred images
        H_mean = np.matmul(np.matmul(K, rotation_mean + np.matmul(translation_mean, norm_v)), np.linalg.inv(K))
        self.H_mean = H_mean/H_mean[2][2]
        self.Hs = Hs

        self.extrinsic_mats = np.array(extrinsic_mats).reshape(self.num_pose+1, 9)
        return Hs
    
        
    def create_blur_image(self, img, Hs):
        # Define the list of the images
        frames = []
        frames.append(img)

        for i in range(self.num_pose):
            h_mat = Hs[i]
            img_dst = cv.warpPerspective(img, h_mat, (self.image_W, self.image_H),flags=cv.INTER_LINEAR+cv.WARP_FILL_OUTLIERS, borderMode=cv.BORDER_REPLICATE)
            frames.append(img_dst)
            
        frames = np.array(frames)
        blur_img = np.mean(frames, axis=0)
        blur_img_rs = self.add_rolling_shutter(blur_img)
        blur_img_rs = blur_img_rs.astype(np.uint8)
        
        return blur_img_rs
        

    def add_rolling_shutter(self, img_blur):
        self.t_readout = self.rng.normal(loc=self.readout_mean, scale=self.readout_std)
        piece_H = int(self.image_H/self.total_sub)
        H_last = self.extrinsic_mats[-1,:]
        H_last = H_last.reshape((3,3))
        K = self.intrinsic_mat

        new_pieces = []
        y = piece_H  # y-1 is the row index

        while y <= self.image_H:
            # time and approximated rotation for y th row
            t_y = self.t_readout * y / self.image_H
            H_y = self.interp_rot(t_y)
            H_new = np.matmul(H_y, np.linalg.inv(H_last))
            W_y = np.matmul(np.matmul(K, H_new), np.linalg.inv(K))

            old_piece = img_blur[y-piece_H:y, :, :]
            new_piece = cv.warpPerspective(old_piece, W_y, (self.image_W, piece_H), flags=cv.INTER_LINEAR+cv.WARP_FILL_OUTLIERS, borderMode=cv.BORDER_REPLICATE)
            new_pieces.append(new_piece)
            y += piece_H

        img_blur_rs = np.concatenate(np.array(new_pieces), axis=0)
        
        return img_blur_rs
        
    def interp_rot(self, t):
        h_array = self.extrinsic_mats
        exposure_ts= np.array([i * self.interval for i in range(self.num_pose+1)])
        if t >= exposure_ts[-1]:
            H_last = h_array[-1, :]
            return H_last.reshape((3,3))

        rot_t = np.array([0.]*9)
        for i in range(9):
            rot_t[i] = np.interp(t, exposure_ts, h_array[:, i])

        return rot_t.reshape((3, 3))
    
    def save_params(self, index):
        # load all params
        blur_params = {}
        blur_params['exposure_time'] = self.exposure_time
        blur_params['readout_time'] = self.t_readout
        blur_params['interval'] = self.interval
        blur_params['num_pose'] = self.num_pose
        blur_params['Hs'] = self.Hs
        blur_params['H_mean'] = self.H_mean
        blur_params['extrinsic_mats'] = self.extrinsic_mats
        blur_params['intrinsic_mat'] = self.intrinsic_mat
        
        np.save(os.path.join(self.output_dir,f"{index+1}.npy"), blur_params)
