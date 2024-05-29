import os
import time
from datetime import date
import socket
import sys

import numpy as np

mapping_grade = {"75bf6":"D",
                "b13a4":"DH",
                "53bfe":"F",
                "d94ec":"FH",
                "12e46":"S",
                "b0a9c":"SH",
                "23aae":"WO",
                "d8c14":"WOH"}

if __name__ == '__main__':
    #
    for TYPE in ['GRADE', 'TUM', 'GRADE_nets']:
        GT_PATH = '/media/ebonetto/WindowsData/SLAM_evaluation'

        GT_TYPE = 'images'
        GT_PATH = os.path.join(GT_PATH, ("GRADE" if TYPE in ['GRADE', 'GRADE_nets'] else "TUM")+f"_{GT_TYPE}")

        if TYPE == 'GRADE':
            path_type_couples = [('/ps/project/irotate/GRADE_new_res/RESULT_SF', 'sf'),
                                 ('/ps/project/irotate/GRADE_new_res/RESULT_DYNASLAM', 'dynaslam'),
                                 ('/ps/project/irotate/GRADE_new_res/RESULT_RTAB', 'rtabmap'),
                                 ('/ps/project/irotate/GRADE_new_res/RESULT_ORB', 'orb'),
                                 ('/ps/project/irotate/GRADE_new_res/RESULTS_DYNAVINS_VO', 'dynavins'),
                                 # ('/ps/project/irotate/GRADE_new_res/RESULTS_DYNAVINS_VIO', 'dynavins'),
                                 ('/ps/project/irotate/GRADE_new_res/RESULT_TARTAN', 'tartan'),
                                 ]

            for TEST_PATH, E_TYPE_SEL in path_type_couples:
                # structure PATH/[3.5,5]/[GT,NOISY]/EXP/logs and bags and .txts
                # structure GT_PATH/[3.5,5]/DATA/EXP/[GT/[depth,rgb], NOISY/[depth,rgb], association.txt, groundtruth.txt] -> GT_TYPE images

                for dist in os.listdir(TEST_PATH):
                    dist_path = os.path.join(TEST_PATH, dist)
                    if "3" in dist:
                        cdist = "3.5"
                    else:
                        cdist = "5"
                    if os.path.isdir(dist_path):
                        for type in os.listdir(dist_path):
                            type_path = os.path.join(dist_path, type)
                            if os.path.isdir(type_path):
                                exps = os.listdir(type_path)
                                exps.sort()
                                for exp in exps:
                                    exp_path = os.path.join(type_path, exp)
                                    if os.path.isdir(exp_path):
                                        print(exp_path)
                                        ename = exp.split('/')[-1]
                                        ename_gt = ''
                                        for k, v in mapping_grade.items():
                                            if k in ename:
                                                ename_gt = v
                                                break

                                        if ename_gt == '':
                                            ename_gt = ename

                                        if GT_TYPE == 'images':
                                            gt_path = os.path.join(GT_PATH, cdist, 'DATA', ename_gt, 'groundtruth.txt')
                                            assoc_path = os.path.join(GT_PATH, cdist, 'DATA', ename_gt, 'association.txt')
                                        else:
                                            gt_path = os.path.join(GT_PATH, cdist, type, ename_gt)

                                        # accumulators
                                        ate = []
                                        miss_time = []
                                        cnt = 0
                                        for file in os.listdir(exp_path):
                                            file_exp = os.path.join(exp_path, file)
                                            if os.path.isdir(file_exp):
                                                continue

                                            if E_TYPE_SEL not in ['dynaslam', 'orb']:
                                                if E_TYPE_SEL == 'rtabmap':
                                                    if '.db' not in file_exp:
                                                        continue
                                                    print(f"Preparing {file_exp}")
                                                    log = os.popen(f"rtabmap-report --poses {file_exp}").read()
                                                    gt_path = file_exp.replace('.db', '_gt.txt')
                                                    file_exp = file_exp.replace('.db', '_slam.txt')
                                                    os.system(f"mv {file_exp} {file_exp.replace(os.path.basename(file_exp), 'CameraTraj_' + os.path.basename(file_exp))}")
                                                    file_exp = file_exp.replace(os.path.basename(file_exp), 'CameraTraj_' + os.path.basename(file_exp))
                                                elif E_TYPE_SEL == 'dynavins':
                                                    if ".bag" not in file_exp:
                                                        continue
                                                    print(f"Preparing {file_exp}")
                                                    log = os.popen(f"python3 ./SLAM_evaluation/src/tool/output_pose.py --type dynavins --path {file_exp} --output {os.path.dirname(file_exp)}").read()
                                                    file_exp = file_exp.replace(os.path.basename(file_exp), 'CameraTraj_'+os.path.basename(file_exp)).replace('.bag', '.txt')
                                                    os.system(f"mv {os.path.join(os.path.dirname(file_exp), 'estimated_pose_dynavins.txt')} {file_exp}")
                                                    gt_path = file_exp.replace('CameraTraj_', '').replace('.txt', '_gt.txt')
                                                    os.system(f"mv {os.path.join(os.path.dirname(file_exp), 'gt_pose_dynavins.txt')} {gt_path}")
                                                elif E_TYPE_SEL == 'tartan':
                                                    if "CameraTraj" not in file_exp or ".png" in file_exp:
                                                        continue
                                                    else:
                                                        print(f"Preparing {file_exp}")
                                                        tartan_name = file_exp.replace("CameraTraj_", "tartan_")
                                                        if os.path.exists(tartan_name):
                                                            print(f"Already processed {file_exp}")
                                                        else:
                                                            os.system(f"mv {file_exp} {tartan_name}")
                                                            log = os.popen(f"python3 ./SLAM_evaluation/src/tool/output_pose.py --type tartan --path {tartan_name} --output {os.path.dirname(file_exp)}").read()
                                                            os.system(f"mv {os.path.join(os.path.dirname(file_exp),'estimated_pose_tartan.txt')} {file_exp}")
                                                elif E_TYPE_SEL == 'sf':
                                                    if "CameraTraj" not in file_exp or ".ply" in file_exp:
                                                        continue
                                                    else:
                                                        print(f"Preparing {file_exp}")
                                                        sf_name = file_exp.replace("CameraTraj_", "sf_")
                                                        if os.path.exists(sf_name):
                                                            print(f"Already processed {file_exp}")
                                                        else:
                                                            os.system(f"mv {file_exp} {sf_name}")
                                                            log = os.popen(
                                                                f"python3 ./SLAM_evaluation/src/tool/output_pose.py --type staticfusion --path {sf_name} --output {os.path.dirname(file_exp)}").read()
                                                            os.system(
                                                                f"mv {os.path.join(os.path.dirname(file_exp), 'estimated_pose_sf.txt')} {file_exp}")

                                                else:
                                                    raise NotImplementedError
                                            else:
                                                if 'CameraTraj' not in file_exp:
                                                    continue
                                            print(f'Processing {file_exp}', end='\r')

                                            log = os.popen(f"python3 ./SLAM_evaluation/src/tool/evaluate_ate.py {gt_path} {file_exp} --verbose"+ " 2>&1").read()
                                            try:
                                                log = log.split('\n')
                                                miss_time_str = log[0]
                                                miss_time.append(float(miss_time_str[miss_time_str.find('Missing')+len('Missing')+1:miss_time_str.find('seconds')-1]))
                                                ate_str = log[2]
                                                ate.append(float(ate_str[ate_str.find('rmse')+len('rmse')+1:ate_str.find(' m')-1]))
                                                cnt += 1
                                            except:
                                                print(f"ERROR WITH {file_exp}")

                                        result_path = os.path.join(exp_path, 'result.txt')

                                        with open(result_path, 'w') as f:
                                            if cnt != 10:
                                                f.write("ERROR")
                                            if cnt == 0:
                                                continue
                                            else:
                                                f.write(f"ATE avg: {np.round(np.mean(ate),3)} std: {np.round(np.std(ate),3)}\n")
                                                f.write(f"MT avg: {np.round(np.mean(miss_time),3)} std: {np.round(np.std(miss_time),3)}\n")
                                                f.write(f"MT: {miss_time}\n")
                                        log = os.popen(f"python3 ./SLAM_evaluation/src/tool/eval.py {gt_path} {assoc_path} {exp_path} 2>&1").read()
                                        log = log.split('\n')[-4:][:3]
                                        new_res_path = os.path.join(exp_path, 'new_res.txt')
                                        with open(new_res_path, 'w') as f:
                                            for line in log:
                                                if line != '':
                                                    f.write(line+'\n')
                                        print(log)
        elif TYPE == "GRADE_nets":
            path_type_couples = []
            for d in os.listdir('/ps/project/irotate/GRADE_new_res/networks_variations/GRADE/DynaSLAM'):
                path_type_couples.append((os.path.join('/ps/project/irotate/GRADE_new_res/networks_variations/GRADE/DynaSLAM',d), 'dynaslam'))
            for d in os.listdir('/ps/project/irotate/GRADE_new_res/networks_variations/GRADE/DynaVINS'):
                path_type_couples.append((os.path.join('/ps/project/irotate/GRADE_new_res/networks_variations/GRADE/DynaVINS',d), 'dynavins'))

            for TEST_PATH, E_TYPE_SEL in path_type_couples:
                exps = os.listdir(TEST_PATH)
                exps.sort()
                for exp in exps: # 23aae, 53bfe, 75bf6, b13a4, d94ec
                    exp_path = os.path.join(TEST_PATH, exp)
                    if not os.path.isdir(exp_path):
                        continue
                    else:
                        print(exp_path)
                        ename = exp.split('/')[-1]
                        ename_gt = ''
                        for k, v in mapping_grade.items():
                            if k in ename:
                                ename_gt = v
                                break

                        if ename_gt == '':
                            ename_gt = ename
                        cdist = "3.5"
                        if GT_TYPE == 'images':
                            gt_path = os.path.join(GT_PATH, cdist, 'DATA', ename_gt, 'groundtruth.txt')
                            assoc_path = os.path.join(GT_PATH, cdist, 'DATA', ename_gt, 'association.txt')
                        else:
                            gt_path = os.path.join(GT_PATH, cdist, type, ename_gt)

                        # accumulators
                        ate = []
                        miss_time = []
                        cnt = 0
                        for file in os.listdir(exp_path):
                            file_exp = os.path.join(exp_path, file)
                            if os.path.isdir(file_exp):
                                continue

                            if E_TYPE_SEL == 'dynavins':
                                if ".bag" not in file_exp:
                                    continue
                                print(f"Preparing {file_exp}")
                                log = os.popen(
                                    f"python3 ./SLAM_evaluation/src/tool/output_pose.py --type dynavins --path {file_exp} --output {os.path.dirname(file_exp)}").read()
                                file_exp = file_exp.replace(os.path.basename(file_exp),
                                                            'CameraTraj_' + os.path.basename(file_exp)).replace('.bag',
                                                                                                                '.txt')
                                os.system(
                                    f"mv {os.path.join(os.path.dirname(file_exp), 'estimated_pose_dynavins.txt')} {file_exp}")
                                gt_path = file_exp.replace('CameraTraj_', '').replace('.txt', '_gt.txt')
                                os.system(
                                    f"mv {os.path.join(os.path.dirname(file_exp), 'gt_pose_dynavins.txt')} {gt_path}")
                            else:
                                if 'CameraTraj' not in file_exp:
                                    continue
                            print(f'Processing {file_exp}', end='\r')

                            log = os.popen(
                                f"python3 ./SLAM_evaluation/src/tool/evaluate_ate.py {gt_path} {file_exp} --verbose" + " 2>&1").read()
                            try:
                                log = log.split('\n')
                                miss_time_str = log[0]
                                miss_time.append(float(miss_time_str[miss_time_str.find('Missing') + len(
                                    'Missing') + 1:miss_time_str.find('seconds') - 1]))
                                ate_str = log[2]
                                ate.append(float(ate_str[ate_str.find('rmse') + len('rmse') + 1:ate_str.find(' m') - 1]))
                                cnt += 1
                            except:
                                print(f"ERROR WITH {file_exp}")

                        result_path = os.path.join(exp_path, 'result.txt')

                        with open(result_path, 'w') as f:
                            if cnt < 10:
                                f.write("ERROR")
                            if cnt == 0:
                                continue
                            else:
                                f.write(f"ATE avg: {np.round(np.mean(ate), 3)} std: {np.round(np.std(ate), 3)}\n")
                                f.write(
                                    f"MT avg: {np.round(np.mean(miss_time), 3)} std: {np.round(np.std(miss_time), 3)}\n")
                        log = os.popen(
                            f"python3 ./SLAM_evaluation/src/tool/eval.py {gt_path} {assoc_path} {exp_path} 2>&1").read()
                        log = log.split('\n')[-4:][:3]
                        new_res_path = os.path.join(exp_path, 'new_res.txt')
                        with open(new_res_path, 'w') as f:
                            for line in log:
                                if line != '':
                                    f.write(line + '\n')
                        print(log)
        else:
            path_type_couples = []
            # for d in os.listdir('/ps/project/irotate/GRADE_new_res/networks_variations/TUM/DynaSLAM'):
            #     path_type_couples.append((os.path.join('/ps/project/irotate/GRADE_new_res/networks_variations/TUM/DynaSLAM',d), 'dynaslam'))
            for d in os.listdir('/ps/project/irotate/GRADE_new_res/networks_variations/TUM/DynaVINS'):
                path_type_couples.append((os.path.join('/ps/project/irotate/GRADE_new_res/networks_variations/TUM/DynaVINS',d), 'dynavins'))

            for TEST_PATH, E_TYPE_SEL in path_type_couples:
                exps = os.listdir(TEST_PATH)
                exps.sort()
                for exp in exps: # halfsphere, rpy, static, xyz
                    exp_path = os.path.join(TEST_PATH, exp)
                    if not os.path.isdir(exp_path):
                        continue
                    else:
                        print(exp_path)
                        ename_gt = exp.split('/')[-1]

                        if GT_TYPE == 'images':
                            gt_path = os.path.join(GT_PATH, ename_gt, 'groundtruth.txt')
                            assoc_path = os.path.join(GT_PATH, ename_gt, f'{ename_gt}.txt')
                        else:
                            raise NotImplementedError

                        # accumulators
                        ate = []
                        miss_time = []
                        cnt = 0
                        for file in os.listdir(exp_path):
                            file_exp = os.path.join(exp_path, file)
                            if os.path.isdir(file_exp):
                                continue

                            if E_TYPE_SEL == 'dynavins':
                                if ".bag" not in file_exp:
                                    continue
                                print(f"Preparing {file_exp}")
                                gt_temp = os.path.join(os.path.dirname(file_exp), 'groundtruth.txt')
                                if not os.path.exists(gt_temp):
                                    os.system(f"cp {gt_path} {gt_temp}")

                                log = os.popen(
                                    f"python3 ./SLAM_evaluation/src/tool/output_pose.py --type dynavins_tum --path {file_exp} --output {os.path.dirname(file_exp)}").read()
                                file_exp = file_exp.replace(os.path.basename(file_exp),
                                                            'CameraTraj_' + os.path.basename(
                                                                file_exp)).replace('.bag', '.txt')
                                os.system(
                                    f"mv {os.path.join(os.path.dirname(file_exp), 'estimated_pose_dynavins.txt')} {file_exp}")
                                gt_path = gt_temp
                            else:
                                if 'CameraTraj' not in file_exp:
                                    continue

                            print(f'Processing {file_exp}', end='\r')

                            log = os.popen(
                                f"python3 ./SLAM_evaluation/src/tool/evaluate_ate.py {gt_path} {file_exp} --verbose --time_thr 0.05" + " 2>&1").read()
                            try:
                                log = log.split('\n')
                                miss_time_str = log[0]
                                miss_time.append(float(miss_time_str[miss_time_str.find('Missing') + len(
                                    'Missing') + 1:miss_time_str.find('seconds') - 1]))
                                ate_str = log[2]
                                ate.append(float(
                                    ate_str[ate_str.find('rmse') + len('rmse') + 1:ate_str.find(' m') - 1]))
                                cnt += 1
                            except:
                                print(f"ERROR WITH {file_exp}")

                        result_path = os.path.join(exp_path, 'result.txt')

                        with open(result_path, 'w') as f:
                            if cnt < 10:
                                f.write("ERROR")
                            if cnt == 0:
                                continue
                            else:
                                f.write(
                                    f"ATE avg: {np.round(np.mean(ate), 3)} std: {np.round(np.std(ate), 3)}\n")
                                f.write(
                                    f"MT avg: {np.round(np.mean(miss_time), 3)} std: {np.round(np.std(miss_time), 3)}\n")
                        log = os.popen(
                            f"python3 ./SLAM_evaluation/src/tool/eval.py {gt_path} {assoc_path} {exp_path} 2>&1").read()
                        log = log.split('\n')[-4:][:3]
                        new_res_path = os.path.join(exp_path, 'new_res.txt')
                        with open(new_res_path, 'w') as f:
                            for line in log:
                                if line != '':
                                    f.write(line + '\n')
                        print(log)