import os
import time
from datetime import datetime
import socket

if __name__ == '__main__':
	DEVICE = socket.gethostname()
	PATH = '/home/ebonetto/GRADE_rgbs'
	OUTPUT = '/home/ebonetto/RESULT_SF/'

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
					print(os.path.join(cexp, f'exp_{idx}'))
					log = os.popen(f"./build/StaticFusion-ImageSeqAssoc {cexp}").read()
					time.sleep(5)

					result_path = os.path.join(OUTPUT, fold, type, exp)
					if not os.path.exists(result_path):
						os.makedirs(result_path)
					t = datetime.now().strftime("%b_%d_%Y_%H%M")
					cam_traj_path = os.path.join(result_path, f'CameraTraj_{str(idx).zfill(3)}_{DEVICE}_{t}')
					os.system(f'mv ./build/sf-mesh.ply {cam_traj_path}.ply')
					os.system(f'mv ./build/sf-mesh.txt {cam_traj_path}.txt')

					with open(os.path.join(result_path, f'log_{str(idx).zfill(3)}_{t}.txt'),'w') as log_file:
						log_file.write(log)
