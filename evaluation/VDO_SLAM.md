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

- Build:
  ```bash
  git clone https://github.com/robot-perception-group/VDO_SLAM.git
  cd VDO_SLAM
  chmod +x build.sh
  ./build.sh
  ```

### 3. Data Preparation

- The organisational structure of the dataset should be:

  ```
  /dataset/rgb/       # folder containing all color images
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

- RGB and depth data can be extracted by 

- The input ground truth camera pose is defined as follows:

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
  ```
- The input ground truth object pose is is organized as follows:

  ```
  FrameID ObjectID B1 B2 B3 B4 t1 t2 t3 r1
  ```

  > Where ti are the coefficients of 3D object location **t** in camera coordinates, and r1 is the Rotation around Y-axis in camera coordinates. B1-4 is 2D bounding box of object in the image, used for visualization.

- Flow data can be generated using [PWC-NET](https://github.com/NVlabs/PWC-Net) with Python 2.7 & PyTorch 0.2 & CUDA 8.0.
  A simple script to process all the data inside a folder and output the flow can be found [here](https://github.com/robot-perception-group/GRADE-eval/tree/main/evaluation/src/script_pwc.py)
- Semantic data is generated using [MASK-RCNN](https://github.com/matterport/Mask_RCNN) with Python 3.6 & tensorflow 1.15.0 & keras 2.2.4 & CUDA 10.2 & cupy-cuda102


### 4. Run

```bash
./example/vdo_slam PARAMETER_YAML_FILE PATH_TO_DATA_FOLDER 1
```

`1` is there because our filenames start from 1.

### 5. Evaluation

- Estimated result will be saved in `refined_stereo_new.txt` in your defined folder
- Evaluate Estimated Result using Absolute Trajecotry Error (ATE) and Trajecotry Plot
  ```bash
  ./evaluate.sh -t vdo -f ESTIMATED_RESULT_TXT (-o OUTPUT) (-s 0.0) (-e 60.0)
  ```
    - `-t|--type` refers to the SLAM method type
    - `-f|--file` refers to the estimated result from VDO-SLAM method
    - `-s|--st` refers to the **start time** for evaluation
    - `-e|--et` refers to the **end time** for evaluation
    - `-o|--od` refers to the output dir. Default to local.
  
### 6. Run your own data

Edit the yaml file accordingly and follow the instructions on the repo page.

If you want you can create a new enum in `include/Tracking.h` and edit accordingly `src/Tracking.cc`.