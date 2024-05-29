import os
import time
from datetime import datetime

import socket

if __name__ == '__main__':
    DEVICE = socket.gethostname()
    PATH = '/home/ebonetto/GRADE_rgbs/GRADE_rgbs_tartan/'
    OUTPUT = '/home/ebonetto/RESULT_DYNASLAM/'

    max_dist_folds = os.listdir(PATH)
    for fold in max_dist_folds:
        if not os.path.isdir(os.path.join(PATH, fold)):
            continue
        print(f"running {fold}")
        for exp in os.listdir(os.path.join(PATH, fold, 'DATA')):
            for type in ['GT', 'NOISY']:
                print(f"running {exp} {type}")
                # import ipdb; ipdb.set_trace()
                cexp = os.path.join(PATH, fold, 'DATA', exp, type)
                for idx in range(10):

                    if os.path.exists('masks'):
                        os.system("rm -r masks/ results/ KeyFrameTrajectory.txt")

                    log = os.popen(f"./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt ./DynaSLAM-mpi.yaml {cexp} ./Examples/RGB-D/associations/rgbd_assoc.txt masks/ results/"+ " 2>&1").read()

                    time.sleep(5)

                    result_path = os.path.join(OUTPUT, fold, type, exp)
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)
                    t = datetime.now().strftime("%b_%d_%Y_%H%M")
                    cam_traj_path = os.path.join(result_path, f'CameraTraj_{str(idx).zfill(3)}_{DEVICE}_{t}.txt')
                    os.system(f'mv CameraTrajectory.txt {cam_traj_path}')

                    with open(os.path.join(result_path, f'log_{str(idx).zfill(3)}_{t}.txt'), 'w') as log_file:
                        log_file.write(log)

                    os.system("rm -r masks/ results/ KeyFrameTrajectory.txt")
                    time.sleep(10)