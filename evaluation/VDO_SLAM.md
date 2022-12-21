## VDO SLAM

### 1. Prerequisties

- C++ 11
- gcc (Test with Version 9.4.0)
- OpenCV (Test with Version 3.4.16)
- Eigen3
- g2o

### 2. Installation

Before installation, please apply the following modification:

- `example/vdo_slam.cc`
    - Change the result saving path at **Line 145** to your desired directory
- `src/Tracking.cc`

    - For RGB-D Depth map, open **Line 193** and **Line 200** as: `imD.at<float>(i,j) = imD.at<float>(i,j)/mDepthMapFactor;`
    - Change **Line 1198** to ignore `mTestData` as: `if (bGlobalBatch)`
    - Change **Line 1668** as: `if (mVelocity.empty() || cur_2d.size() == 0)`
    - Change **Line 1654** as the following block:

  ```
  try{
      cv::solvePnPRansac(pre_3d, cur_2d, camera_mat, distCoeffs, Rvec, Tvec, \
      false, iter_num, reprojectionError, confidence, inliers, cv::SOLVEPNP_AP3P);
      }
  catch(cv::Exception){
      cout << "rvec:" << Rvec << endl;
      cout << "tvec:" << Tvec << endl;
      }
  ```

- Build:
  ```bash
  git clone https://github.com/halajun/VDO_SLAM.git VDO-SLAM
  cd VDO-SLAM
  chmod +x build.sh
  ./build.sh
  ```

### 3. Run

```bash
./example/vdo_slam PARAMETER_YAML_FILE PATH_TO_DATA_FOLDER
```

> Download customized parameter yaml file for VDO-SLAM from [here](https://github.com/Kyle-Xu001/Synthetic-Robotic-Data-Generation/blob/main/launch/config/vdo-mpi.yaml).

### 4. Data Preparation

- The organisational structure of the dataset should be:

  ```
  /dataset/image_0/   # folder containing all color images
  /dataset/depth/     # folder containing all depth images
  /dataset/flow/      # folder containing all the estimated flow files
  /dataset/semantic/  # folder containing all semantic labels for each frame
  /dataset/object_pose.txt # Ground Truth Object Pose
  /dataset/pose_gt.txt # Ground Truth Camera Pose
  /dataset/times.txt
  ```

- The expected format of the images:
    - Color images - 8 bit RGB Image in PNG
    - Depth images - 16 bit monochrome in PNG, scaled by 5000 (**Defined in Yaml File**)
- The input ground truth camera pose is organized as follows:

  ```
  FrameID R11 R12 R13 t1 R21 R22 R23 t2 R31 R32 R33 t3 0 0 0 1
  ```

  > Here Rij are the coefficients of the camera rotation matrix **R** and ti are the coefficients of the camera translation vector **t**.

- Generate `pose_gt.txt` requried by VDO-SLAM by running:
  ```bash
  # generate gt_pose_vdo.txt
  python3 src/tool/output_pose.py --type vdo_gt \
                                  --path REINDEX_BAG_SEQUENCE \
                                  --topic /my_robot_0/camera/pose
  mv gt_pose_vdo.txt pose_gt.txt # rename pose groundtruth file
  ```
- The input ground truth object pose is is organized as follows:

  ```
  FrameID ObjectID B1 B2 B3 B4 t1 t2 t3 r1
  ```

  > Where ti are the coefficients of 3D object location **t** in camera coordinates, and r1 is the Rotation around Y-axis in camera coordinates. B1-4 is 2D bounding box of object in the image, used for visualization.

- Flow data can be generated using [PWC-NET](https://github.com/NVlabs/PWC-Net) with Python 2.7 & PyTorch 0.2 & CUDA 8.0.
- Semantic data is generated using [MASK-RCNN](https://github.com/matterport/Mask_RCNN) with Python 3.6 & tensorflow 1.15.0 & keras 2.2.4 & CUDA 10.2 & cupy-cuda102

### 5. Evaluation

- Estimated result will be saved in `refined_stereo_new.txt` in your defined folder
- Evaluate Estimated Result using Absolute Trajecotry Error (ATE) and Trajecotry Plot
  ```bash
  ./evaluate.sh -t vdo -f ESTIMATED_RESULT_TXT (-s 0.0) (-e 60.0)
  ```
    - `-t|--type` refers to the SLAM method type
    - `-f|--file` refers to the estimated result from VDO-SLAM method
    - `-s|--st` refers to the **start time** for evaluation
    - `-e|--et` refers to the **end time** for evaluation