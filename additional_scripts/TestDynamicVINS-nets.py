import os
import time
import roslaunch
import rospy
import sys
from datetime import datetime

models = [

#("/../model/a_grade.pt", 0, []),
#("/../model/a_grade_coco.pt", 0, []),
#("/../model/a-grade-coco-mixed.pt", 0, []),
#("/../model/A-GRADE-S-COCO-MIX.pt", 0, []),
#("/../model/a_grade_s_coco.pt", 0, []),
#("/../model/s_coco.pt", 0, []),
("/../model/S-GRADE-COCO-MIX.pt", 0, []),
("/../model/s_grade_coco.pt", 0, []),
("/../model/s_grade.pt", 10, []),
#("/../model/S-GRADE-S-COCO-MIX.pt", 0, []),
("/../model/s_grade_s_coco.pt", 10, []),
("/../model/yolov5s.pt", 0, []),
]

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)
    print(*args, flush=True, **kwargs)

def replace_line(
    new_content,
    file_path="/home/cxu/Project/DynaVINS_ws/src/Dynamic-VINS/yolo_ros/src/yolo_bridge/yolo_bridge.py",
    line_number=48,
    keyword="+",
):
    with open(file_path, "r") as file:
        lines = file.readlines()
    cl = lines[line_number - 1]
    idx = cl.find(keyword)
    new_line = cl[: idx + len(keyword)] + new_content + "\n"
    lines[line_number - 1] = new_line
    with open(file_path, "w") as file:
        file.writelines(lines)


if __name__ == "__main__":
    DEVICE = "ps043"
    PATH = "/home/cxu/GRADE_bags/GRADE_bags/3.5/GT"
    OUTPUT = "/home/cxu/RESULT_ELIA_DYNAVINS/"

    os.system("rm ~/.ros -rf")

    for m, NUM_EXP, todo in models:
        MODEL = m.split("/")[-1].split(".pt")[0]
        replace_line(new_content=f'"{m}",')

        for i in range(NUM_EXP):
            for exp in os.listdir(PATH):
                if len(todo) > 0:
                    if exp not in todo:
                        continue
                eprint("===============================================================")
                eprint("===============================================================")
                eprint("===============================================================")
                eprint(f"RUNNING {i} over {NUM_EXP} for {exp} and {MODEL}")
                eprint("===============================================================")
                eprint("===============================================================")
                eprint("===============================================================")
                # start testing
                rospy.init_node("VINS", anonymous=True)
                uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
                roslaunch.configure_logging(uuid)
                
                cli_args = [
                    "src/Dynamic-VINS/vins_estimator/launch/tum_rgbd/grade_pytorch.launch",
                    f"path:={PATH}",f"exp:={exp}","config:=tum_fr3_mpi.yaml"]
                
                roslaunch_args = cli_args[1:]
                roslaunch_file = [
                (roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
                
                launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
                launch.start()
                
                rospy.loginfo("started")
                while True:
                    ttt = os.popen('ps -ax | grep rosbag | grep play').readlines()
                    time.sleep(10)
                    if len(ttt) == 1:
                        time.sleep(10)
                        break
                
                launch.shutdown()
                time.sleep(30)

#                ttt = os.popen('ps -ax | grep ros').readlines()
#                for proc in ttt:
#                    if not 'master' in proc and not 'core' in proc and not:
#                        id_proc = (int) proc.split(' ')[0]
#                        os.popen(f'kill -9 {id_proc}')

                # create testing folder
                result_path = os.path.join(OUTPUT,MODEL,exp)
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    
                time.sleep(10)
                t = datetime.now()
                
                t = t.strftime("%b-%d-%Y_%H%M")
                
                # copy estimated trajectory
                os.system(
                    f"mv ~/.ros/{exp}*.bag {result_path}/%s_{DEVICE}_{t}.bag"
                    % str(i + 1).zfill(3))
                os.system(
                    f"mv ~/.ros/log {result_path}/logs_%s_{DEVICE}_{t}"
                    % str(i + 1).zfill(3))
