import os
import time
import roslaunch
import rospy
import sys
from datetime import datetime

import socket

if __name__ == "__main__":
	DEVICE = socket.gethostname()
	PATH = "/home/cxu/GRADE_bags/GRADE_bags"
	OUTPUT = "/home/cxu/RESULTS_DYNAVINS/"

	os.system("rm ~/.ros -rf")

	for dist in os.listdir(PATH):
		if not os.path.isdir(os.path.join(PATH, dist)):
			continue
		for type in ['GT', 'NOISY']:
			fold = os.path.join(PATH, dist, type)
			for exp in os.listdir(fold):
				for i in range(10):
#					import ipdb; ipdb.set_trace()

					# start testing
					rospy.init_node("VINS", anonymous=True)
					uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
					roslaunch.configure_logging(uuid)

					cexp = os.path.join(PATH, dist, type, exp)
					cli_args = [
						"src/Dynamic-VINS/vins_estimator/launch/tum_rgbd/grade_pytorch.launch",
						f"path:={os.path.join(PATH, dist, type)}", f"exp:={exp}", f"config:=tum_fr3_mpi{'.yaml' if '3' in dist else '_5m.yaml'}"]

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
							time.sleep(30)
							break

					launch.shutdown()
					time.sleep(30)


					# create testing folder
					result_path = os.path.join(OUTPUT,dist,type,exp)
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
