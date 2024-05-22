## ORB-SLAM2

### 1. Prerequisties

- Ubuntu 20.04
- C++ 11
- gcc/g++ (Test with Version 4.8)
- Eigen 3
- [Pangolin-0.5](https://github.com/robot-perception-group/Pangolin)
- OpenCV (Test with Version 3.4.16)

#### Installing gcc/g++ 4.8

- Modify `sources.list`

  ```bash
  sudo gedit /etc/apt/sources.list
  ```

    - Add following lines to the end

  ```
  deb http://dk.archive.ubuntu.com/ubuntu/ xenial main
  deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe
  ```

- Installation

  ```bash
  sudo apt update
  sudo apt install g++-4.8 gcc-4.8 -y
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 10
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 10
  ```

- Change gcc and g++ version to 4.8
  ```bash
  sudo update-alternatives --config gcc
  sudo update-alternatives --config g++
  ```

#### Installing Pangolin-0.5 (with **gcc/g++ 4.8**)

```bash
sudo apt install libglew-dev cmake libpython2.7-dev
# clone modified pangolin repo
git clone https://github.com/robot-perception-group/Pangolin.git
cd Pangolin
mkdir build && cd build
cmake .. # to install locally [-D CMAKE_INSTALL_PREFIX=../install]
make -j && sudo make install
```

### 2.Installation

- Compile ORB-SLAM2 with **gcc/g++ 4.8**

  ```bash
  sudo apt install libboost-all-dev
  git clone https://github.com/raulmur/ORB_SLAM2.git ORB_SLAM2
  cd ORB_SLAM2
  chmod +x build.sh
  ```
  - Change **Line 50** of `include/LoopClosing.h` as:

  ```cpp
  Eigen::aligned_allocator<std::pair<KeyFrame *const, g2o::Sim3> > > KeyFrameAndPose;
  ```

  - Add following line to the head of `include/Viewer.h`:

  ```cpp
  #include <unistd.h>
  ```
  - Run `./build.sh`


### 3. Run

- Execute the following command. Change `PATH_TO_SEQUENCE_FOLDER` to the sequence folder.

  ```
  ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt PARAMETER_YAML_FILE PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
  ```

  > - Download customized parameter yaml file from [here](https://github.com/Kyle-Xu001/Synthetic-Robotic-Data-Generation/blob/main/launch/config/DynaSLAM-mpi.yaml).
  > - Download association file from [here](https://github.com/Kyle-Xu001/Synthetic-Robotic-Data-Generation/blob/main/config/rgbd_assoc.txt) to associate corresponding color and depth images.

- The organisational structure of the dataset should be:
  ```
  /dataset/rgb/   # folder containing all color images
  /dataset/depth/ # folder containing all depth images
  /dataset/rgbd_assoc.txt # association file
  ```
- The expected format of the images:
    - Color images - 8 bit in PNG. Resolution: VGA
    - Depth images - 16 bit monochrome in PNG, scaled by 1000. Resolution: VGA

### 4. Evaluation

- Estimated result will be saved in `CameraTrajectory.txt` at `ORB_SLAM2` folder
- Evaluate Estimated Result using Absolute Trajectory Error (ATE), Relative Pose Error (RPE) and Trajecotry Plot
  ```bash
  ./evaluate.sh -t orbslam2 -f ESTIMATED_RESULT_TXT -g GROUNTRUTH_BAG_FOLDER (-o OUTPUTDIR) (-s 0.0) (-e 60.0)
  ```
    - `-t|--type` refers to the SLAM method type
    - `-f|--file` refers to the estimated result from ORB-SLAM2 method
    - `-s|--st` refers to the **start time** for evaluation
    - `-e|--et` refers to the **end time** for evaluation
    - `-o|--od` refers to the output directory
    - `-g|--gb` refers to the folder of grountruth bags from which we extract `gt_pose.txt`
