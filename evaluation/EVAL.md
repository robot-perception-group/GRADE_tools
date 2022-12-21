## EVALUATION
This document will guide you to the installation of all the SLAM/V(I)O systems that were used in the evaluation of the GRADE paper.
Moreover, we will extract groundtruth information necessary during the evaluation step.

### Extract Ground Truth Pose Data

`gt_pose.txt` is required by multiple SLAM methods. Run following command to generate desired gt-pose data.

We leave to the user where to generate those files and how to run the following evaluations.

- Default Ground Truth Pose File
  ```bash
  python3 src/tool/output_pose.py --type groundtruth \
                                  --path BAG_SEQUENCE_FOLDER[reindex_bags_folder] \
                                  --topic /my_robot_0/camera/pose
  ```
- VDO-SLAM requried Ground Truth Pose File `gt_pose_vdo.txt`
  ```bash
  # generate gt_pose_vdo.txt
  python3 src/tool/output_pose.py --type vdo_gt \
                                  --path BAG_SEQUENCE_FOLDER[reindex_bags_folder] \
                                  --topic /my_robot_0/camera/pose
  mv gt_pose_vdo.txt pose_gt.txt # Rename pose groundtruth file
  ```
- Tartan VO required Ground Truth Pose File `gt_pose_tartan.txt` by transforming **Default `gt_pose.txt`**
  ```bash
  python3 src/tool/output_pose.py --type tartan_gt --path gt_pose.txt
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