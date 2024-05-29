## EVALUATION
This document will guide you to the installation of all the SLAM/V(I)O systems that were used in the evaluation of the GRADE paper.
Moreover, we will extract groundtruth information necessary during the evaluation step.

Everything has been tested in 
- UBUNTU 20.04
- ROS NOETIC

Please refer to [https://wiki.ros.org/noetic/Installation/Ubuntu](https://wiki.ros.org/noetic/Installation/Ubuntu) to install ROS.

#### NOTE:
Experiments have been run by using a `roscore` and setting `use_sim_time` true.

To install libraries locally (suggested) you can use `-D CMAKE_INSTALL_PREFIX=../install`.
Remember that at that point each program using that library needs to be manually linked through the `CMakeLists.txt` file, usually specifying `PATH ...` in the `find_package` calls.

As per the python packages, we leave to the reader the choice if they want to run them in virtual environments or not.

### Extract Ground Truth Pose Data

`gt_pose.txt` is required by multiple SLAM methods. Run following command to generate desired gt-pose data.


#### NOTE, if --od is left empty the default is the local one. Be aware of possible overwritings.
- Default Ground Truth Pose File
  ```bash
  python3 src/tool/output_pose.py --type groundtruth \
                                  --path BAG_SEQUENCE_FOLDER[reindex_bags_folder] \
                                  --topic /my_robot_0/camera/pose
                                  --output OUTPUT_DIR
  ```
- VDO-SLAM requried Ground Truth Pose File `gt_pose_vdo.txt`
  ```bash
  # generate gt_pose_vdo.txt
  python3 src/tool/output_pose.py --type vdo_gt \
                                  --path BAG_SEQUENCE_FOLDER[reindex_bags_folder] \
                                  --topic /my_robot_0/camera/pose
                                  --output OUTPUT_DIR
  mv gt_pose_vdo.txt pose_gt.txt # Rename pose groundtruth file
  ```
- Tartan VO required Ground Truth Pose File `gt_pose_tartan.txt` by transforming **Default `gt_pose.txt`**
  ```bash
  python3 src/tool/output_pose.py --type tartan_gt \
                                  --path gt_pose.txt \ # default groundtruth pose
                                  --output OUTPUT_DIR
  ```

### EVALUATION INSTALLATION INSTRUCTIONS
Each link contains all the edits that we made (or the forks that we used), and how to run evaluations for each project.
Most of the edits were necessary to run the projects in the latest Ubuntu version and address many of the difficulties that we encountered during testing.
Some of them were made to avoid crashes of the selected methods, especially when no features were detected.

Some bulk-test scripts can be found either in `SLAM_evaluation/additional_scripts` or in the specific forked repositories.

- [RTAB-Map](https://github.com/introlab/rtabmap_ros.git) from [Multi-Session Visual SLAM for Illumination-Invariant Re-Localization in Indoor Environments](https://arxiv.org/abs/2103.03827)
- [Dynamic-VINS](https://github.com/robot-perception-group/Dynamic-VINS) from [RGB-D Inertial Odometry for a Resource-Restricted Robot in Dynamic Environments](https://ieeexplore.ieee.org/document/9830851/)
- [VDO SLAM](https://github.com/robot-perception-group/VDO_SLAM) from [VDO-SLAM: A Visual Dynamic Object-aware SLAM System](https://arxiv.org/abs/2005.11052)
- [TartanVO](https://github.com/robot-perception-group/tartanvo) from [TartanVO: A Generalizable Learning-based VO](https://arxiv.org/pdf/2011.00359.pdf)
- [DynaSLAM](https://github.com/BertaBescos/DynaSLAM) from [DynaSLAM: Tracking, Mapping, and Inpainting in Dynamic Scenes](https://ieeexplore.ieee.org/document/8421015)

### RUN ON YOUR OWN DATA
Instructions are specified in each singular tutorial.

If you need to change the evaluation, the `evaluate.sh` is where you want to look. We extract the ground truth trajectory from the bags in most cases (except for RTABMap which log that automatically from the TF tree).

You can easily skip that by running your evaluation by hand using pre-fixed files.

### ADD ADDITIONAL METHODS
Please feel free to evaluate on different methods and, by following the same scheme, write a tutorial and a pull-request. We will happily include them here.
The evaluation script is modular based on the `-t` flag that selects the method.
