import os
import time
import roslaunch
import rospy
import sys
from datetime import datetime
import socket

if __name__ == '__main__':
        DEVICE = socket.gethostname()
        PATH = '/home/ebonetto/GRADE_bags'
        OUTPUT = '/home/ebonetto/RESULT_RTAB/'


        max_dist_folds = os.listdir(PATH)
        for fold in max_dist_folds:
                if not os.path.isdir(os.path.join(PATH, fold)):
                        continue
                print(f"running {fold}")
                for type in os.listdir(os.path.join(PATH, fold)):
                        for exp in os.listdir(os.path.join(PATH, fold, type)):
                                print(f"running {exp} {type}")
                                cexp = os.path.join(PATH, fold, type, exp)
                                for idx in range(10):

                                    os.system("rm ~/.ros -rf")
                                    rospy.init_node("RTAB", anonymous=True)
                                    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
                                    roslaunch.configure_logging(uuid)
                                    if 's' in exp.lower():
                                        rgb_topic = '/my_robot_0/camera_link/0/rgb/image_raw' 
                                        depth_topic = '/my_robot_0/camera_link/0/depth/image_raw'
                                        camera_topic = '/my_robot_0/camera_link/0/camera_info'
                                    else:
                                        rgb_topic = '/my_robot_0/camera_link/1/rgb/image_raw' 
                                        depth_topic = '/my_robot_0/camera_link/1/depth/image_raw'
                                        camera_topic = '/my_robot_0/camera_link/1/camera_info'
                                    cli_args = [
                                               "src/rtabmap_ros/rtabmap_ros/launch/rgbdslam_datasets.launch",
                    f"rgb_topic:={rgb_topic}",f"depth_topic:={depth_topic}",f"camera_info_topic:={camera_topic}", f"exp:={exp}", f"path:={os.path.join(PATH, fold, type)}"]
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

                                    # create testing folder
                                    result_path = os.path.join(OUTPUT,fold,type,exp)
                                    if not os.path.exists(result_path):
                                        os.makedirs(result_path)
                                    time.sleep(10)
                                    t = datetime.now()
                                    t = t.strftime("%b-%d-%Y_%H%M")
                                    os.system(f"mv ~/.ros/*.db {result_path}/%s_{DEVICE}_{t}.db"% str(idx + 1).zfill(3))
                                    os.system(f"mv ~/.ros/log {result_path}/logs_%s_{DEVICE}_{t}"%str(idx + 1).zfill(3))
