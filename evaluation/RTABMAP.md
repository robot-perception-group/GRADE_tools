## RTAB-Map

### 1. Prerequisites

- Ubuntu 20.04
- OpenCV-3.4.16 (**with non-free modules**)
- ROS Noetic

### 2. Installation

- Set up your ROS workspace with:
  ```bash
  source /opt/ros/noetic/setup.bash
  cd ~/catkin_ws/src # Your workspace
  catkin_init_workspace
  ```
- Install Required Dependencies:
  ```bash
  sudo apt install ros-noetic-rtabmap ros-noetic-rtabmap-ros
  sudo apt remove ros-noetic-rtabmap ros-noetic-rtabmap-ros
  sudo apt install ros-noetic-libg2o
  ```
- Install RTAB-Map Standalone Libraries **outside your ROS workspace**:

  ```bash
  cd ~
  git clone https://github.com/introlab/rtabmap.git rtabmap
  cd rtabmap/build
  cmake ..
  make -j
  sudo make install
  ```

- Install RTAB-Map ROS Package in your ROS workspace:
  ```bash
  cd ~/catkin_ws
  git clone https://github.com/introlab/rtabmap_ros.git src/rtabmap
  catkin_make
  source devel/setup.bash
  ```

### 3. Run

- Download provided `rtabmap_test.launch` from [here](https://github.com/Kyle-Xu001/Synthetic-Robotic-Data-Generation/blob/main/launch/rtabmap_test.launch) to your `rtabmap_ros/launch` folder.
- Launch RTAB-Map Node:
  ```bash
  roslaunch rtabmap_ros rtabmap_test.launch
  ```
- Play rosbags in another terminal:
  ```bash
  rosparam set use_sim_time True
  rosbag play /dataset/*.bag --clock
  ```

### 4. Evaluation

- Evaluate Estimated Result using Absolute Trajecotry Error (ATE) and Trajecotry Plot
  ```bash
  ./evaluate.sh -t rtabmap (-s 0.0) (-e 60.0)
  ```
    - `-t|--type` refers to the SLAM method type
    - `-s|--st` refers to the **start time** for evaluation
    - `-e|--et` refers to the **end time** for evaluation
- (Optional) Output the Absolute Trajecotry Error (ATE) using the following command:
  ```bash
  rtabmap-report --poses ~/.ros/rtabmap.db
  ```
  > The estimated pose `rtabmap_slam.txt` and ground truth pose `rtabmap_gt.txt` results will be saved in `~/.ros`.
