import os
import time
from datetime import datetime
import socket

if __name__ == '__main__':
	DEVICE = socket.gethostname()
	PATH = '/home/ebonetto/GRADE_rgbs_tartan'
	OUTPUT = '/home/ebonetto/RESULT_ORB/'

	max_dist_folds = os.listdir(PATH)
	for fold in max_dist_folds:
		if not os.path.isdir(os.path.join(PATH, fold)):
			continue
		print(f"running {fold}")
		for exp in os.listdir(os.path.join(PATH, fold, 'DATA')):
			for type in ['GT', 'NOISY']:
				print(f"running {exp} {type}")
				cexp = os.path.join(PATH, fold, 'DATA', exp, type)
				for idx in range(10):
					if os.path.exists('/home/ebonetto/ORBSLAM/ORB_SLAM2/KeyFrameTrajectory.txt'):
						os.system("rm -r /home/ebonetto/ORBSLAM/ORB_SLAM2/KeyFrameTrajectory.txt")
					if os.path.exists('/home/ebonetto/ORBSLAM/ORB_SLAM2/CameraTrajectory.txt'):
						os.system("rm -r /home/ebonetto/ORBSLAM/ORB_SLAM2/CameraTrajectory.txt")
					print(os.path.join(cexp, f'exp_{idx}'))
					log = os.popen(f"./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt DynaSLAM-mpi.yaml {cexp} {cexp}/../association.txt").read()
					time.sleep(5)

					result_path = os.path.join(OUTPUT, fold, type, exp)
					if not os.path.exists(result_path):
						os.makedirs(result_path)
					t = datetime.now().strftime("%b_%d_%Y_%H%M")
					cam_traj_path = os.path.join(result_path, f'CameraTraj_{str(idx).zfill(3)}_{DEVICE}_{t}.txt')
					os.system(f'mv ~/ORBSLAM/ORB_SLAM2/CameraTrajectory.txt {cam_traj_path}')
					with open(os.path.join(result_path, f'log_{str(idx).zfill(3)}_{t}.txt'),'w') as log_file:
						log_file.write(log)
"""            
    for m, NUM_EXP, exclude in models:
        MODEL = m.split('/')[1].split('_best')[0]
        print(f"running {MODEL}")
        replace_line(new_content = f"\"{m}\")")

        for i in range(NUM_EXP):
            print(f"runnging {i}/{NUM_EXP}", end = '\r')
            for exp in os.listdir(PATH):
                if len(exclude):
                    if exp in exclude:
                        continue
                if os.path.exists('masks'):
                    os.system("rm -r masks/ results/ KeyFrameTrajectory.txt")
                exp_path = os.path.join(PATH, exp)
                log = os.popen(f"./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt ./Examples/RGB-D/TUM3.yaml {exp_path} ./Examples/RGB-D/associations/fr3_walking_{exp}.txt masks/ results/"+ " 2>&1").read()
                time.sleep(15)

                result_path = os.path.join(OUTPUT, 'DynaSLAM', MODEL, exp)
                if not os.path.exists(result_path):
                    os.makedirs(result_path)

                t = datetime.now().strftime("%b_%d_%Y_%H%M")
                cam_traj_path = os.path.join(result_path, f'CameraTraj_{str(i+1).zfill(3)}_{DEVICE}_{t}.txt')
                os.system(f'mv CameraTrajectory.txt {cam_traj_path}')
                with open(os.path.join(result_path, f'log_{str(i+1).zfill(3)}_{t}.txt'),'w') as log_file:
                    log_file.write(log)
                os.system("rm -r masks/ results/ KeyFrameTrajectory.txt")
                time.sleep(10)
"""
