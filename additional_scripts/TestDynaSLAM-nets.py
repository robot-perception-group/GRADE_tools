import os
import time
from datetime import datetime

models = [
('detectron2_models/A-GRADE_best_segm.pth',0,['static','xyz']),
('detectron2_models/A-GRADE+COCO_best_segm.pth',0,[]),
('detectron2_models/A-GRADE+COCO_mixed_best_segm.pth',0,[]),
('detectron2_models/A-GRADE+S-COCO_best_segm.pth',2,['rpy','static','xyz']),
('detectron2_models/A-GRADE+S-COCO_mixed_best_segm.pth',0,[]),
('detectron2_models/COCO_best_segm.pth',0,[]),
('detectron2_models/S-COCO_best_segm.pth',0,[]),
('detectron2_models/S-GRADE+COCO_best_segm.pth',0,[]),
('detectron2_models/S-GRADE+COCO_mixed_best_segm.pth',0,[]),
('detectron2_models/S-GRADE+S-COCO_best_segm.pth',1,['rpy','static','xyz']),
('detectron2_models/S-GRADE+S-COCO_mixed_best_segm.pth',0,[]),
('detectron2_models/S-GRADE_best_segm.pth',0,[])]

def replace_line(new_content, file_path = '/home/ebonetto/SLAM_TEST/DynaSLAM/src/python/MaskRCNN.py', line_number = 145, keyword = 'ROOT_DIR,'):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    cl = lines[line_number - 1]
    idx = cl.find(keyword)
    new_line = cl[:idx+len(keyword)] + new_content + '\n'
    lines[line_number - 1] = new_line
    with open(file_path, 'w') as file:
        file.writelines(lines)


if __name__ == '__main__':
    DEVICE = 'ps043'
    PATH = '/home/ebonetto/TUM_RGBD'
    OUTPUT = '/home/ebonetto/RESULT_ELIA_0.9_NMS_0.3/'
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
