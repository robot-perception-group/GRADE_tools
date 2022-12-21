## EVALUATION
This document will guide you to the installation of all the SLAM/V(I)O systems that were used in the evaluation of the GRADE paper.
Moreover, we will extract groundtruth information necessary during the evaluation step.

Everything has been tested in 
- UBUNTU 20.04
- ROS NOETIC

Please refer to [https://wiki.ros.org/noetic/Installation/Ubuntu](https://wiki.ros.org/noetic/Installation/Ubuntu) to install ROS.

#### NOTE:
Experiments have been run by using a `roscore` and setting `use_sim_time` true.

### Extract Ground Truth Pose Data

`gt_pose.txt` is required by multiple SLAM methods. Run following command to generate desired gt-pose data.

We leave to the user where to generate those files and how to run the following evaluations.

#### NOTE, if --od is left empty the default is the local one. Be aware of possible overwritings.
- Default Ground Truth Pose File
  ```bash
  python3 src/tool/output_pose.py --type groundtruth \
                                  --path BAG_SEQUENCE_FOLDER[reindex_bags_folder] \
                                  --topic /my_robot_0/camera/pose
                                  --od output_dir
  ```
- VDO-SLAM requried Ground Truth Pose File `gt_pose_vdo.txt`
  ```bash
  # generate gt_pose_vdo.txt
  python3 src/tool/output_pose.py --type vdo_gt \
                                  --path BAG_SEQUENCE_FOLDER[reindex_bags_folder] \
                                  --topic /my_robot_0/camera/pose
                                  --od output_dir
  mv gt_pose_vdo.txt pose_gt.txt # Rename pose groundtruth file
  ```
- Tartan VO required Ground Truth Pose File `gt_pose_tartan.txt` by transforming **Default `gt_pose.txt`**
  ```bash
  python3 src/tool/output_pose.py --type tartan_gt --path gt_pose.txt --od output_dir
  ```

### EVALUATION INSTALLATION INSTRUCTIONS
Each link contains all the edits that we made (or the forks that we used), and how to run evaluations for each project.
Most of the edits were necessary to run the projects in the latest Ubuntu version and address many of the difficulties that we encountered during testing.
Some of them were made to avoid crashes of the selected methods, especially when no features were detected.

- [RTAB-Map]() from [title]()
- [Dynamic-VINS]() from [title]()
- [VDO SLAM]() from [title]()
- [TartanVO]() from [title]()
- [DynaSLAM]() from [title]()


# CX-TODO ADD LINKS + publication/arxiv

### RUN ON YOUR OWN DATA
Instructions are specified in each singular tutorial.

### ADD ADDITIONAL METHODS
Please feel free to evaluate on different methods and, by following the same scheme, write a tutorial and a pull-request. We will happily include them here.
The evaluation script is modular based on the `-t` flag that selects the method.