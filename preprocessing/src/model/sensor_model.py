# Python
import numpy as np
import matplotlib.pyplot as plt
from struct import pack, unpack


class IMU:
    def __init__(self, angular_velocity, linear_acceleration, time):
        self.linear_acceleration = linear_acceleration
        self.angular_velocity = angular_velocity
        self.time = time


class Image:
    def __init__(self, depth, color = None, pointcloud_enable = False):
        self.color_image = color
        self.depth_image = depth
        self.pointcloud_enable = pointcloud_enable


class SensorModel:
    def __init__(self, type, noise_model, config, seed):
        '''  Initialize ros node and read params '''

        self.seed = seed
        self.rng = np.random.default_rng(seed = self.seed)
        
        # Read in params
        if type in ['imu', 'camera', 'pointcloud']:
            self.sensor_type = type
            
            if noise_model == 'groundtruth':
                self.noise_model = 'groundtruth'
                
            elif type == 'camera' or type == 'pointcloud':
                if noise_model in ['gaussian_depth_noise', 'kinect', 'realsense']:
                    self.noise_model = noise_model
                    self.maximum_depth = config['maximum_depth']
                    self.maximum_depth_variance = config['maximum_depth_variance']
                    self.subpixel = config['subpixel']
                    self.flatten_distance = config['flatten_distance']

                    if self.noise_model == 'gaussian':
                        # coefficients for polynomial, f(z) = k0 + k1z + k2z^2 + k3z^3
                        self.coefficients = np.array([0.0] * 8)
                        for i in range(4):
                            self.coefficients[i] = config['k_mu_%i' % i][i]
                            self.coefficients[4 + i] = config['k_sigma_%i' % i][i]
                            
                    if len(config['camera_params']) == 6:
                        self.camera_params = config['camera_params']  # [width, height, fx, fy, cx, cy]
                    elif len(config['camera_params']) == 3:
                        self.camera_params = config['camera_params'] + [config['camera_params'][-1],
                                                                        config['camera_params'][0] / 2,
                                                                        config['camera_params'][1] / 2]
                    else:
                        raise ValueError('Invalid camera params. \n'
                                         'Need to be either [width, height, fx, fy, cx, cy] or [w, h, f]')
                    self.hfov = np.deg2rad(config['hfov'])  # horizontal field of view in radians
                else:
                    raise ValueError('Invalid noise model for camera')
                
            elif type == 'imu':
                self.noise_model = 'gaussian'
                self.frequnecy = config['frequency']
                self.gyro_bias_corr_time = config['gyro_bias_corr_time']
                self.gyro_noise_density = config['gyro_noise_density']
                self.gyro_random_walk = config['gyro_random_walk']
                self.accel_bias_corr_time = config['accel_bias_corr_time']
                self.accel_noise_density = config['accel_noise_density']
                self.accel_random_walk = config['accel_random_walk']
                self.gyro_bias = np.zeros(3)
                self.accel_bias = np.zeros(3)
                self.gyro_turn_bias = self.rng.uniform(0, config['gyro_turn_bias'], 3)
                self.accel_turn_bias = self.rng.uniform(0, config['accel_turn_bias'], 3)
                self.prev_t = -1
        else:
            raise ValueError('Invalid sensor type')


    def callback(self, data):
        ''' Produce simulated sensor outputs from data '''
        if self.noise_model == 'groundtruth':
            return data
        elif self.sensor_type == 'imu':
            return self.process_imu(data)
        elif self.sensor_type == 'camera':
            return self.process_camera(data)
            
    def process_camera(self, data):
        # Read out images
        if self.noise_model == "kinect":
            img_color = data.color_image
            # Pack RGB image (for ros representation)
            rgb = self.rgb_to_float(img_color)

        img_depth = data.depth_image

        # Build 3D point cloud from depth
        if self.flatten_distance > 0:
            img_depth = np.clip(img_depth, 0, self.flatten_distance)
        
        # Transform Depth Image to Points
        (x, y, z) = self.depth_to_3d(img_depth)        

        # Sensor model processing, put other processing functions here
        if self.noise_model == 'kinect':
            # NOTE THAT KINECT SHOULD BE TREATED AS PCL + RGB (so noise also in x and y)
            x, y, z, rgb =  self.process_kinect(x, y, z, rgb)
        
        elif self.noise_model == 'gaussian':
            z = self.process_gaussian_depth_noise(z)
            depth = np.array(z).reshape(self.camera_params[1],self.camera_params[0])
        
        elif self.noise_model == 'realsense':
            z = self.process_realsense(z)
            depth = np.array(z).reshape(self.camera_params[1],self.camera_params[0])
        
        # Remove invalid points
        if self.maximum_depth > 0:
            mask = depth > self.maximum_depth
            depth[mask] = 0
            
            if data.pointcloud_enable == True:
                mask = z <= self.maximum_depth
                x = x[mask]
                y = y[mask]
                z = z[mask]
                if self.noise_model == "kinect":
                    rgb = rgb[mask]
                
        return x, y, z, depth
        
    def depth_to_3d(self, img_depth):
        ''' Create point cloud from depth image and camera params. Returns a single array for x, y and z coords '''
        # read camera params and create image mesh
        width = self.camera_params[0]
        height = self.camera_params[1]
        fx = self.camera_params[2]
        fy = self.camera_params[3]
        center_x = self.camera_params[4]
        center_y = self.camera_params[5]

        cols, rows = np.meshgrid(np.linspace(0, width - 1, num=width), np.linspace(0, height - 1, num=height))

        # Create x and y position
        points_x = img_depth * (cols - center_x) / fx
        points_y = img_depth * (rows - center_y) / fy

        return points_x.reshape(-1), points_y.reshape(-1), img_depth.reshape(-1)

    @staticmethod
    def rgb_to_float(img_color):
        ''' Stack uint8 rgb image into a single float array (efficiently) for ros compatibility '''
        r = np.ravel(img_color[:, :, 0]).astype(int)
        g = np.ravel(img_color[:, :, 1]).astype(int)
        b = np.ravel(img_color[:, :, 2]).astype(int)
        color = np.left_shift(r, 16) + np.left_shift(g, 8) + b
        packed = pack('%di' % len(color), *color)
        unpacked = unpack('%df' % len(color), packed)
        return np.array(unpacked)

    def process_kinect(self, x, y, z, rgb):
        # Simulate a kinect sensor model according to this paper: https://ieeexplore.ieee.org/abstract/document/6375037
        # The effect of the target plane orientation theta is neglected (its constant until theta ~ 60/70 deg).
        mask = (z > 0.5) & (z < 3.0)
        x = x[mask]
        y = y[mask]
        z = z[mask]
        rgb = rgb[mask]
        sigma_l = 0.0014 * z
        sigma_z = 0.0012 + 0.0019 * (z - 0.4) ** 2
        mu = np.zeros(np.shape(z))
        dx = self.rng.normal(mu, sigma_l)
        dy = self.rng.normal(mu, sigma_l)
        dz = self.rng.normal(mu, sigma_z)
        return x + dx, y + dy, z + dz, rgb

    def process_realsense(self, z):
        # Simulate a realsense sensor model according to this https://www.intel.com/content/dam/support/us/en/documents/emerging-technologies/intel-realsense-technology/BKMs_Tuning_RealSense_D4xx_Cam.pdf
        f = 0.5 * self.camera_params[0] / np.tan(self.hfov / 2)
        multiplier = self.subpixel / (f * 0.05)
        rms_noise = (z) ** 2 * multiplier

        z_noise = z + self.rng.normal(0, np.minimum(rms_noise, self.maximum_depth_variance))
        z_noise = z_noise.astype(np.float32)
        
        '''Debug'''
        # count = []
        # noises = []
        # interval = np.linspace(0, 4, 201)
        
        # for i in interval:
        #     mask = (z >= i) & (z < i+0.2)
        #     noises.append(np.mean(rms_noise[mask]))
        #     count.append(np.count_nonzero((rms_noise>=i) & (rms_noise<i+0.2)))
        
        # plt.figure(0)
        # plt.bar(interval, count, width = 0.02)
        # plt.xlabel("Noise Varience Value")
        # plt.ylabel("Count")
        
        # plt.figure(1)
        # plt.bar(interval, noises, width = 0.02)
        # plt.ylabel("Noise Varience")
        # plt.xlabel("Depth Distance")
        # plt.show()
        
        # index = np.abs(z-3).argmin() 
        # print("f:",f)
        # print("multiplier:", multiplier)
        # print("rms_noise:", rms_noise[index])
        # print("normal_noise", self.rng.normal(0, np.minimum(rms_noise, self.maximum_depth_variance))[index],"\n\n")
        
        return z_noise

    def process_gaussian_depth_noise(self, z_in):
        # Add a depth dependent guassian error term to the perceived depth. Mean and stddev can be specified as up to
        # deg3 polynomials.
        mu = np.ones(np.shape(z_in)) * self.coefficients[0]
        sigma = np.ones(np.shape(z_in)) * self.coefficients[4]
        for i in range(1, 4):
            if self.coefficients[i] != 0:
                mu = mu + np.power(z_in, i) * self.coefficients[i]
            if self.coefficients[4 + i] != 0:
                sigma = sigma + np.power(z_in, i) * self.coefficients[4 + i]
        return z_in + self.rng.normal(mu, sigma)

    def process_imu(self, data):
        # Read out IMU data
        ang_vel_data = data.angular_velocity
        accel_data = data.linear_acceleration
        
        if self.prev_t == -1:
            self.prev_t = data.time
            return ang_vel_data, accel_data
        
        # dt = data.time - self.prev_t
        # if dt > 0:
        #     self.prev_t = data.time
        # else:
        #     dt = 1 / self.frequnecy
        #     print("Warning: IMU data is not in order [Current Time: %s, Previous Time: %s]"%(data.time, self.prev_t))
        #     return ang_vel_data, accel_data
        dt = 1 / self.frequnecy
        
        tau_g = self.gyro_bias_corr_time
        sigma_g_d = 1 / (dt ** 0.5) * self.gyro_noise_density
        sigma_b_g = self.gyro_random_walk
        sigma_b_g_d = (-sigma_b_g ** 2 * tau_g / 2 * (np.exp(-2 * dt / tau_g) - 1)) ** 0.5

        phi_g_d = np.exp(-1 / tau_g * dt)
        
        self.gyro_bias = phi_g_d * self.gyro_bias + self.rng.normal(0, sigma_b_g_d, 3)
        [ang_vel_data.x, ang_vel_data.y, ang_vel_data.z] = [ang_vel_data.x, ang_vel_data.y, ang_vel_data.z] + self.gyro_bias + self.rng.normal(0, sigma_g_d, 3) + self.gyro_turn_bias
        
        tau_a = self.accel_bias_corr_time
        sigma_a_d = 1 / (dt ** 0.5) * self.accel_noise_density
        sigma_b_a = self.accel_random_walk
        sigma_b_a_d = (-sigma_b_a ** 2 * tau_a / 2 * (np.exp(-2 * dt / tau_a) - 1)) ** 0.5
        phi_a_d = np.exp(-1 / tau_a * dt)
        
        self.accel_bias = phi_a_d * self.accel_bias + self.rng.normal(0, sigma_b_a_d, 3)
        [accel_data.x, accel_data.y, accel_data.z] = [accel_data.x, accel_data.y, accel_data.z] + self.accel_bias + self.rng.normal(0, sigma_a_d, 3) + self.accel_turn_bias

        return ang_vel_data, accel_data
