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

- Generate `pose_gt.txt` and `times.txt` requried by VDO-SLAM by running:
  ```bash
  # generate gt_pose_vdo.txt
  python3 src/tool/output_pose.py --type vdo_gt \
                                  --path REINDEX_BAG_SEQUENCE \
                                  --topic /my_robot_0/camera/pose
  ```
  remember to move the scripts output eventually (or use `-d`).
  
- The input ground truth object pose is organized as follows:

  ```
  FrameID ObjectID B1 B2 B3 B4 t1 t2 t3 r1
  ```

  Where ti are the coefficients of 3D object location **t** in camera coordinates, and r1 is the Rotation around Y-axis in camera coordinates. B1-4 is 2D bounding box of object in the image, used for visualization.
  
  If you do not have it you can use an empty file.
  
- Define the output folder [here](https://github.com/robot-perception-group/VDO_SLAM/blob/master/example/vdo_slam.cc#L145) and build VDO again.

- Flow data can be generated using [PWC-NET](https://github.com/NVlabs/PWC-Net) with Python 2.7 & PyTorch 0.2 & CUDA 8.0.
  A simple script to process all the data inside a folder and output the flow can be found [here](https://github.com/robot-perception-group/GRADE-eval/tree/main/evaluation/src/script_pwc.py)

- Semantic data can be generated using [MASK-RCNN](https://github.com/matterport/Mask_RCNN) with Python 3.6 & tensorflow 1.15.0 & keras 2.2.4 & CUDA 10.2 & cupy-cuda102. **Note** remember to install `pycocotools` as explained in MaskRCNN readme and to `install` it (not just make it, see [here](https://github.com/matterport/Mask_RCNN/issues/1595)).
  We provide a notebook [here](https://github.com/robot-perception-group/GRADE-eval/blob/main/evaluation/src/maskrcnn/demo.ipynb) and a script [here](https://github.com/robot-perception-group/GRADE-eval/blob/main/evaluation/src/maskrcnn/mask_gen.py) (run it as `./mask_gen.py input_rgb_folder out_semantic_folder`) to generate the data. Both need to be copied in the main MaskRCNN folder. 
  The modified visualization script can be found [here](https://github.com/robot-perception-group/GRADE-eval/blob/main/evaluation/src/maskrcnn/visualize.py), please copy it to `Mask_RCNN\mrcnn\visualize.py` and then install Mask_RCNN.
  If you want you can use newer python versions according your own system. If you have issues with tensorflow/keras when you want to use the newer version you may need to check [this](https://github.com/matterport/Mask_RCNN/issues/2075#issuecomment-905845632).
  

### 4. Run

```bash
./example/vdo_slam PARAMETER_YAML_FILE PATH_TO_DATA_FOLDER OFFSET_INITIAL_FILENAME
```

`OFFSET_...` is there in case your filenames starts from an index different from 0. Use `0`, `1` ....

The original `VDO_SLAM\example\vdo_slam.cc` considers KITTI filename convention. To change this please change [this](https://github.com/robot-perception-group/VDO_SLAM/blob/master/example/vdo_slam.cc#L188) line and run `build.sh` again.

YAML file can be found in `GRADE-eval/evaluation/config/vdo-mpi.yaml`

### 5. Evaluation

- Estimated result will be saved in `refined_stereo_new.txt` in your defined folder [here](https://github.com/robot-perception-group/VDO_SLAM/blob/master/example/vdo_slam.cc#L145).
  
- Evaluate Estimated Result using Absolute Trajectory Error (ATE) and Trajecotry Plot
  ```bash
  ./evaluate.sh -t vdo -f ESTIMATED_RESULT_TXT -g GT_BAGS (-o OUTPUT) (-s 0.0) (-e 60.0)
  ```
    - `-t|--type` refers to the SLAM method type
    - `-f|--file` refers to the estimated result from VDO-SLAM method
    - `-s|--st` refers to the **start time** for evaluation
    - `-e|--et` refers to the **end time** for evaluation
    - `-o|--od` refers to the output dir. Default to local.
    - `-g|--gb` refers to the grountruth bags from which we extract gt_pose.
  
### 6. Run your own data

Edit the yaml file accordingly and follow the instructions on the repo page.

If you want you can create a new enum in `include/Tracking.h` and edit accordingly `src/Tracking.cc`.