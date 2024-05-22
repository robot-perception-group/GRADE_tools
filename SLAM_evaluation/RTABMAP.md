## RTAB-Map

### 1. Prerequisites

- Ubuntu 20.04
- OpenCV-3.4.16 (**with non-free modules**), optional if building from source
- ROS Noetic
- RTAB_ROS was used at the commit 5024574e4ab8df16d3b9d9731ea86761fbbeeb54, RTABMAP at 467dc738c707c7dc6370a03f39df541cf9347b3e. Without GTSAM.
<details>
  <code>
    -- Info :
--   RTAB-Map Version =     0.20.22
--   CMAKE_VERSION =        3.16.3
--   CMAKE_INSTALL_PREFIX = /usr/local
--   CMAKE_BUILD_TYPE =     Release
--   CMAKE_INSTALL_LIBDIR = lib
--   BUILD_APP =            ON
--   BUILD_TOOLS =          ON
--   BUILD_EXAMPLES =       ON
--   BUILD_SHARED_LIBS =    ON
--   CMAKE_CXX_FLAGS =  -fmessage-length=0  -fopenmp -std=c++14
--   FLANN_KDTREE_MEM_OPT = OFF
--   PCL_DEFINITIONS = -DDISABLE_OPENNI2;-DDISABLE_PCAP;-DDISABLE_PNG;-DDISABLE_LIBUSB_1_0
--   PCL_VERSION = 1.10.0
--
-- Optional dependencies ('*' affects some default parameters) :
--  *With OpenCV 4.2.0 xfeatures2d = NO, nonfree = NO (License: BSD)
--   With Qt 5.12.8            = YES (License: Open Source or Commercial)
--   With VTK 7.1              = YES (License: BSD)
--   With external SQLite3     = YES (License: Public Domain)
--   With ORB OcTree           = YES (License: GPLv3)
--   With SupertPoint          = NO (WITH_TORCH=OFF)
--   With Python3              = NO (WITH_PYTHON=OFF)
--   With Madgwick             = YES (License: GPL)
--   With FastCV               = NO (FastCV not found)
--   With PDAL                 = NO (PDAL not found)
--
--  Solvers:
--   With TORO                 = YES (License: Creative Commons [Attribution-NonCommercial-ShareAlike])
--  *With g2o                  = YES (License: BSD)
--  *With GTSAM                = NO (WITH_GTSAM=OFF)
--  *With Ceres                = NO (WITH_CERES=OFF)
--   With VERTIGO              = YES (License: GPLv3)
--   With cvsba                = NO (WITH_CVSBA=OFF)
--  *With libpointmatcher      = YES (License: BSD)
--   With CCCoreLib            = NO (WITH_CCCORELIB=OFF)
--   With Open3D               = NO (WITH_OPEN3D=OFF)
--   With OpenGV               = NO (WITH_OPENGV=OFF)
--
--  Reconstruction Approaches:
--   With OCTOMAP              = YES (License: BSD)
--   With CPUTSDF              = NO (WITH_CPUTSDF=OFF)
--   With OpenChisel           = NO (WITH_OPENCHISEL=OFF)
--   With AliceVision          = NO (WITH_ALICE_VISION=OFF)
--
--  Camera Drivers:
--   With Freenect             = NO (libfreenect not found)
--   With OpenNI2              = YES (License: Apache v2)
--   With Freenect2            = NO (libfreenect2 not found)
--   With Kinect for Windows 2 = NO (Kinect for Windows 2 SDK not found)
--   With Kinect for Azure     = NO (Kinect for Azure SDK not found)
--   With dc1394               = YES (License: LGPL)
--   With FlyCapture2/Triclops = NO (Point Grey SDK not found)
--   With ZED                  = NO (ZED sdk and/or cuda not found)
--   With ZEDOC                = NO (ZED Open Capture not found)
--   With RealSense            = NO (librealsense not found)
--   With RealSense2           = YES (License: Apache-2)
--   With MyntEyeS             = NO (mynteye s sdk not found)
--   With DepthAI              = NO (WITH_DEPTHAI=OFF)
--
--  Odometry Approaches:
--   With loam_velodyne        = NO (WITH_LOAM=OFF)
--   With floam                = NO (WITH_FLOAM=OFF)
--   With libfovis             = NO (WITH_FOVIS=OFF)
--   With libviso2             = NO (WITH_VISO2=OFF)
--   With dvo_core             = NO (WITH_DVO=OFF)
--   With okvis                = NO (WITH_OKVIS=OFF)
--   With msckf_vio            = NO (WITH_MSCKF_VIO=OFF)
--   With VINS-Fusion          = NO (WITH_VINS=OFF)
--   With OpenVINS             = NO (WITH_OPENVINS=OFF)
--   With ORB_SLAM             = NO (WITH_ORB_SLAM=OFF)
-- Show all options with: cmake -LA | grep WITH_
  </code>
</details>

### 2. Installation

- Set up your ROS workspace with:
  ```bash
  source /opt/ros/noetic/setup.bash
  mkdir -p ~/catkin_ws/src  # Your workspace
  cd ~/catkin_ws/src 
  catkin_init_workspace
  ```
- Install from apt **this limits a bit the usage**:
  ```bash
  sudo apt install ros-noetic-rtabmap ros-noetic-rtabmap-ros
  ```
- Install from source **outside your ROS workspace**:
  
  You can customize this however you want. E.g. installing GTSAM, nonfree opencv libraries, libpointmatcher etc.
  Follow [this](https://github.com/eliabntt/irotate_active_slam/blob/noetic/INSTALL.md#:~:text=follow%20this%20README-,Install%20GTSAM4,-.x%20either%20from) for more insight.
  ```bash
  sudo apt install ros-noetic-rtabmap ros-noetic-rtabmap-ros
  sudo apt remove ros-noetic-rtabmap ros-noetic-rtabmap-ros
  git clone https://github.com/introlab/rtabmap.git rtabmap # outside catkin
  cd rtabmap/build
  cmake ..
  make -j
  sudo make install
  ```

- Install RTAB-Map ROS Package in your catkin workspace:
  ```bash
  cd your_catkin_ws/src
  git clone https://github.com/introlab/rtabmap_ros.git # we used commit 5024574e4ab8df16d3b9d9731ea86761fbbeeb54
  cd ..
  catkin_make
  source devel/setup.bash
  ```

- Copy/move the `GRADE_tools/SLAM_evaluation/launch/rgbdslam_datasets.launch` in `your_catkin_ws/src/rtabmap_ros/launch/`

### 3. Run

Assuming roscore is running and `use_sim_time` is `True`, as we will be playing rosbags.
Please edit the parameters of  `rgbdslam_datasets.launch` (the topics) according to your needs.
- Launch RTAB-Map Node:
  ```bash
  roslaunch rtabmap_ros rgbdslam_datasets.launch
  ```
- Play rosbags in another terminal:
  ```bash
  rosbag play /dataset/*.bag --clock
  ```

### 4. Evaluation
By default `rtabmap_ros` will save the database is `.ros/rtabmap.db`.

Please change it in the config file or move it after running the experiment if you desire something different.

- Evaluate Estimated Result using Absolute Trajectory Error (ATE), Relative Pose Error (RPE) and Trajecotry Plot
  ```bash
  ./evaluate.sh -t rtabmap -f .ros/rtabmap.db -o OUTPUT_DIR (-s 0.0) (-e 60.0)
  ```
    - `-t|--type` refers to the SLAM method type
    - `-f|--file` refers to the RTABMAP database
    - `-o|--od` refers to the output dir. Default to `.`
    - `-s|--st` refers to the **start time** for evaluation
    - `-e|--et` refers to the **end time** for evaluation

  The resulting files will be placed in the `output_dir` if specified. You can find the `rtabmap_gt.txt` `rtabmap_slam.txt` and the plot.

### 5. Run on your own data
Simply change the topics in the launch file and follow plenty of instructions of RTABMap official repo.
