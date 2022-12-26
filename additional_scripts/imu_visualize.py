import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_dir = ""
    imu_dir = ""
    imu_noisy_dir = ""
    
    '''Run once and Load the data from error.npy file'''
    files = os.listdir(data_dir)
    if "error.npy" not in files or "imu.npy" not in files:
        imu_files = os.listdir(imu_dir)
        imu_files.sort(key=lambda x:(int(x.split('-')[0][4:]),int(x.split('-')[1][4:-4])))
        
        noise_error = np.zeros((len(imu_files),6))
        ts = np.zeros((len(imu_files),))
        index = 0
        
        imus = np.zeros((len(imu_files),6))
        for imu_file in imu_files:
            with open(os.path.join(imu_dir, imu_file),'rb') as f:
                imu = np.load(f, allow_pickle=True)
                
            with open(os.path.join(imu_noisy_dir, imu_file),'rb') as f:
                imu_noisy = np.load(f, allow_pickle=True)
            
            # Load Raw IMU data
            imus[index, 0] = imu.item()["ang_vel"][0]
            imus[index, 1] = imu.item()["ang_vel"][1]
            imus[index, 2] = imu.item()["ang_vel"][2]
            imus[index, 3] = imu.item()["lin_acc"][0]
            imus[index, 4] = imu.item()["lin_acc"][1]
            imus[index, 5] = imu.item()["lin_acc"][2]
            
            # Load noisy IMU difference
            noise_error[index, 0] = imu_noisy.item()["ang_vel"][0] - imu.item()["ang_vel"][0]
            noise_error[index, 1] = imu_noisy.item()["ang_vel"][1] - imu.item()["ang_vel"][1]
            noise_error[index, 2] = imu_noisy.item()["ang_vel"][2] - imu.item()["ang_vel"][2]
            noise_error[index, 3] = imu_noisy.item()["lin_acc"][0] - imu.item()["lin_acc"][0]
            noise_error[index, 4] = imu_noisy.item()["lin_acc"][1] - imu.item()["lin_acc"][1]
            noise_error[index, 5] = imu_noisy.item()["lin_acc"][2] - imu.item()["lin_acc"][2]
            
            ts[index] = imu_noisy.item()["time"]
            
            index += 1
            
        '''Save the Data'''
        noise_data = {}
        noise_data["noise_error"] = noise_error
        noise_data["ts"] = ts
        
        np.save(os.path.join(data_dir, "error.npy"), noise_data)
        np.save(os.path.join(data_dir, "imu.npy"), imus)
    else:
        noise_data = np.load(os.path.join(data_dir, "error.npy"), allow_pickle=True)
        noise_error = noise_data.item()["noise_error"]
        imus = np.load(os.path.join(data_dir, "imu.npy"), allow_pickle=True)
        ts = noise_data.item()["ts"]
    
    '''Polyfit the Data'''
    ang_x = np.polyfit(ts, noise_error[:,0], 1)
    ang_y = np.polyfit(ts, noise_error[:,1], 1)
    ang_z = np.polyfit(ts, noise_error[:,2], 1)
    acc_x = np.polyfit(ts, noise_error[:,3], 1)
    acc_y = np.polyfit(ts, noise_error[:,4], 1)
    acc_z = np.polyfit(ts, noise_error[:,5], 1)
    p_ang_x = np.poly1d(ang_x)
    p_ang_y = np.poly1d(ang_y)
    p_ang_z = np.poly1d(ang_z)
    p_acc_x = np.poly1d(acc_x)
    p_acc_y = np.poly1d(acc_y)
    p_acc_z = np.poly1d(acc_z)
    
    '''Plot Error Trend Image'''
    plt.figure(0)
    plt.plot(ts, p_ang_x(ts), ts, p_ang_y(ts), ts, p_ang_z(ts))
    plt.title("Angular Velocity Noise Trend")
    plt.legend(['Angular Velocity X', 'Angular Velocity Y', 'Angular Velocity Z'])
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Error (rad/s)")
    
    
    plt.figure(1)
    plt.plot(ts, p_acc_x(ts), ts, p_acc_y(ts), ts, p_acc_z(ts))
    plt.title("Linear Acceleration Noise Trend")
    plt.legend(['Linear Acceleration X', 'Linear Acceleration Y', 'Linear Acceleration Z'])
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Error (m/s)")
    
    '''Plot Error Image'''
    plt.figure(2)
    plt.plot(ts, noise_error[:,0], ts, noise_error[:,1], ts, noise_error[:,2])
    plt.title("Angular Velocity Noise")
    plt.legend(['Angular Velocity X', 'Angular Velocity Y', 'Angular Velocity Z'])
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Error (rad/s)")
    
    plt.figure(3)
    plt.plot(ts, noise_error[:,3], ts, noise_error[:,4], ts, noise_error[:,5])
    plt.title("Linear Acceleration Noise")
    plt.legend(['Linear Acceleration X', 'Linear Acceleration Y', 'Linear Acceleration Z'])
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Error (m/s)")
    
    '''Plot Original Image'''
    plt.figure(4)
    plt.plot(ts, imus[:,0], ts, imus[:,1], ts, imus[:,2])
    plt.title("Original Angular Velocity Data")
    plt.legend(['Angular Velocity X', 'Angular Velocity Y', 'Angular Velocity Z'])
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Angular Velocity (rad/s)")
    
    plt.figure(5)
    plt.plot(ts, imus[:,3], ts, imus[:,4], ts, imus[:,5])
    plt.title("Original Linear Acceleration Data")
    plt.legend(['Linear Acceleration X', 'Linear Acceleration Y', 'Linear Acceleration Z'])
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Linear Acceleration (m/s)")
    plt.show()
        