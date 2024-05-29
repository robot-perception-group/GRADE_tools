import os
import time
from datetime import datetime
import socket

if __name__ == '__main__':
	DEVICE = socket.gethostname()
	PATH = '/home/ebonetto/GRADE_rgbs_tartan'
	OUTPUT = '/home/ebonetto/RESULT_TARTAN/'

	max_dist_folds = os.listdir(PATH)
	for fold in max_dist_folds:
		if not os.path.isdir(os.path.join(PATH, fold)):
			continue
		print(f"running {fold}")
		for exp in os.listdir(os.path.join(PATH, fold, 'DATA')):
			if exp != "D-75bf6":
				continue
			for type in ['GT', 'NOISY']:
				if type != "GT":
					continue
				print(f"running {exp} {type}")
				cexp = os.path.join(PATH, fold, 'DATA', exp, type)
				for idx in range(10):
					if os.path.exists('/home/ebonetto/tartanvo/results'):
						os.system("rm -r /home/ebonetto/tartanvo/results")
					os.makedirs('/home/ebonetto/tartanvo/results')
					print(os.path.join(cexp, f'exp_{idx}'))
					if not os.path.exists(os.path.join(cexp, 'gt_pose_tartan.txt')):
						log = os.popen(f"cp {os.path.join(PATH, fold, 'DATA', exp, 'groundtruth.txt')} {os.path.join(cexp, 'gt_pose.txt')}").read()
						log = os.popen(f"python3 ~/GRADE_tools/SLAM_evaluation/src/tool/output_pose.py --type tartan_gt --path {os.path.join(cexp, 'gt_pose.txt')} --output {cexp}").read()
					log = os.popen(f"python vo_trajectory_from_folder.py --grade --model-name tartanvo_1914.pkl \
                                    --test-dir {os.path.join(cexp,'rgb')} \
                                    --pose-file {os.path.join(cexp,'gt_pose_tartan.txt')} \
                                    --batch-size 1 \
                                    --worker-num 1").read()
					time.sleep(5)

					result_path = os.path.join(OUTPUT, fold, type, exp)
					if not os.path.exists(result_path):
						os.makedirs(result_path)
					t = datetime.now().strftime("%b_%d_%Y_%H%M")
					cam_traj_path = os.path.join(result_path, f'CameraTraj_{str(idx).zfill(3)}_{DEVICE}_{t}.txt')
					os.system(f'mv ~/tartanvo/results/grade_tartanvo_1914.txt {cam_traj_path}')
					os.system(f'mv ~/tartanvo/results/grade_tartanvo_1914.png {cam_traj_path[:-4]+".png"}')
					with open(os.path.join(result_path, f'log_{str(idx).zfill(3)}_{t}.txt'),'w') as log_file:
						log_file.write(log)
					os.system("rm -r ~/tartanvo/results/")
					break
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
