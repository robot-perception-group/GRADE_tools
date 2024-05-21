## StaticFusion
### 1. Prerequisites

- Ubuntu 20.04
- OpenCV (Test with Version 3.4.16)
- CUDA (Test with Version 10.2)
- [Pangolin-0.8](https://github.com/stevenlovegrove/Pangolin/releases/tag/v0.8)

#### Installing MRPT

```bash
sudo add-apt-repository ppa:joseluisblancoc/mrpt-stable
sudo apt install libmrpt-dev mrpt-apps
```

#### Install Other Dependencies

```bash
sudo apt install cmake freeglut3-dev libglew-dev libopencv-dev libopenni2-dev git
```

#### Install Pangolin
```bash
wget https://github.com/stevenlovegrove/Pangolin/archive/refs/tags/v0.8.zip
unzip v0.8.zip
rm v0.8.zip
cd Pangolin-0.8
mkdir build
cd build
cmake .. # to install locally [-D CMAKE_INSTALL_PREFIX=../install]
make
[sudo] make install
```

### 2. Installation

- Clone and checkout branch 
  ```
  git clone git@github.com:christian-rauch/staticfusion.git
  cd staticfusion
  git checkout updates # Switch to this updated branch
  ```
- Modify the **Camera Parameters** at **Line 124** of `StaticFusion.h` as following **before installation**:
  ```cpp
  StaticFusion(unsigned int width = 640 / 2, 
               unsigned int height = 480 / 2, 
               float fx =  325.910278 / 2, 
               float fy = 399.310333 / 2, 
               float cx = 320 / 2., 
               float cy = 240 / 2.);
  ```
  These are the width, heigth... of the resized images, irrespective of the input size.

  As of Version 2.12.0: Released March 17th, 2024 of MRPT, you will also need to change
  ```cpp
  Utils/Datasets.cpp
  // from
  timestamp_obs = mrpt::system::timestampTotime_t(obs3D->timestamp);
  // to
  timestamp_obs = mrpt::Clock::toDouble(obs3D->timestamp); 
  ```
  
- Build
  ```bash
  mkdir build & cd build
  cmake ..
  make
  ```

### 3. Run

- The organisational structure of the dataset should be:
  ```
  /dataset/rgb/   # folder containing all color images
  /dataset/depth/ # folder containing all depth images
  /dataset/rgbd_assoc.txt
  ```
  You can extract the rgb and depth by running [this](https://github.com/robot-perception-group/GRADE-eval/blob/main/preprocessing/PREPROCESSING.md#2-data-extraction)


- The expected format of the images:
    - Color images - 8 bit in PNG.
    - Depth images - 16 bit monochrome in PNG, scaled by 1000.


- `rgbd_assoc.txt` contain a list of items of the form as shown below, which can be found in `GRADE-eval/SLAM_evaluation/config/rgbd_assoc.txt` to associate corresponding color and depth images.
  ```
  timestamp1 /rgb/rgb_id.png timestamp2 /depth/depth_id.png
  ```
  You have to copy that file to the corresponding dataset folder. 
  

- #### USE `./StaticFusion-ImageSeqAssoc PATH_TO_DATASET` [from the build folder]
    - Run `sudo ldconfig` if error occurs when loading shared libraries: `libpango_windowing.so`
    - Click **SAVE** button after completing entire experiment to generate **estimated pose** `sf-mesh.txt` and **pointcloud** `st-mesh.ply` results in `/build` folder.

### 4.Evaluation

- Evaluate Estimated Result using Absolute Trajectory Error (ATE), Relative Pose Error (RPE) and Trajecotry Plot
  ```bash
  ./evaluate.sh -t staticfusion -f sf-mesh.txt -g GROUNTRUTH_BAG_FOLDER (-o OUTPUT_DIR) (-s 0.0) (-e 60.0)
  ```
    - `-t|--type` refers to the SLAM method type
    - `-f|--file` refers to the estimated result from Tartan VO method
    - `-o|--od` refers to the output_dir. If not specified `.` is used
    - `-s|--st` refers to the **start time** for evaluation
    - `-e|--et` refers to the **end time** for evaluation
    - `-g|--gb` refers to the folder of grountruth bags from which we extract `gt_pose.txt`

### 5. Running on your own data

Edit the `StaticFusion.h` accordingly.
Generate the association file through the TUM-RGBD associate file you can find in `GRADE-eval/SLAM_evaluation/src/tool/associate.py`.
