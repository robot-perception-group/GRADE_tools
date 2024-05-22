## RTAB-Map

### 1. Prerequisites

- Ubuntu 20.04
- OpenCV-3.4.16 (**with non-free modules**), optional if building from source
- ROS Noetic

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
  git clone https://github.com/introlab/rtabmap_ros.git
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
