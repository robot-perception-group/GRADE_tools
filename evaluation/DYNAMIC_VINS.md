## Dynamic-VINS

### 1. Prerequisties

- ROS Noetic:
  ```bash
  sudo apt install ros-noetic-cv-bridge ros-noetic-tf ros-noetic-message-filters ros-noetic-image-transport ros-noetic-nav-msgs ros-noetic-visualization-msgs
  ```
- Ceres-Solver: (Similar to the Official Instruction [here](https://github.com/HITSZ-NRSL/Dynamic-VINS/blob/main/doc/INSTALL.md))
  ```bash
  sudo apt install cmake
  sudo apt install libgoogle-glog-dev libgflags-dev
  sudo apt install libatlas-base-dev
  sudo apt install libeigen3-dev
  sudo apt install libsuitesparse-dev
  ```
  ```bash
  git clone https://ceres-solver.googlesource.com/ceres-solver
  cd ceres-solver
  git checkout 2.0.0
  mkdir ceres-bin
  cd ceres-bin
  cmake ..
  make -j
  sudo make install
  ```
- Sophus: (Similar to the Official Instruction [here](https://github.com/HITSZ-NRSL/Dynamic-VINS/blob/main/doc/INSTALL.md))

  ```bash
  git clone https://github.com/strasdat/Sophus.git
  cd Sophus
  git checkout a621ff  # revert
  ```

  `gedit sophus/so2.cpp`: Modify `sophus/so2.cpp` as

  ```
  SO2::SO2()
  {
  unit_complex_.real(1.0);
  unit_complex_.imag(0.0);
  }
  ```

  Build Sophus:

  ```bash
  mkdir build && cd build && cmake .. && sudo make install
  ```

### 2. Installation

```bash
cd catkin_ws/src
# clone modified fork
git clone git@github.com:robot-perception-group/Dynamic-VINS.git

# install python dependencies
sudo apt install ros-noetic-ros-numpy
sudo apt install python3-pip
pip3 install --upgrade pip
conda create -n dvins python=3.6
conda activate dvins
# install pytorch with CUDA based on your device (or conda install)
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install -r yolo_ros/requirements.txt

# build
cd ..
catkin_make
source devel/setup.bash
```

- Change Python Path at **Line 1** of `yolo_ros/src/demo_service_server.py` to your conda environment.
- Make sure that packages `camera_model` and `vins_estimator` are using the OpenCV from ROS (version 4.2) instead of other OpenCV built from source, otherwise it will fail when loading `/EstimatorNodelet`.

### 3. Run

- Visual Inertial Odometry:
  ```bash
  roslaunch vins_estimator openloris_vio_pytorch_mpi.launch \
                  rgb:="/my_robot_0/camera_link/0/rgb/image_raw" \
                  depth:="/my_robot_0/camera_link/0/depth/image_raw" \
                  imu:="/my_robot_0/imu_body"
  roslaunch vins_estimator vins_rviz.launch
  rosbag play /dataset/*.bag --clock
  ```
- Visual Odometry:
  `bash
  roslaunch vins_estimator tum_rgbd_pytorch_mpi.launch \
  rgb:="/my_robot_0/camera_link/0/rgb/image_raw" \
  depth:="/my_robot_0/camera_link/0/depth/image_raw"
  roslaunch vins_estimator vins_rviz.launch
  rosbag play /dataset/*.bag --clock
  `
  > - Start playing bags after `vins_estimator` and `yolo_ros` nodes are initialized
  > - Estimation process may be delayed. Slowing down the rosbag play speed may improve it.
- Save Estimated Result during experiments:
  ```bash
  rosbag record /my_robot_0/camera/pose \
              /vins_estimator/camera_pose \
              /vins_estimator/init_map_time -O result.bag
  ```

### 4. Evaluation

- Evaluate Estimated Result using Absolute Trajecotry Error (ATE) and Trajecotry Plot
  ```bash
  ./evaluate.sh -t dynavins -f result.bag (-s 0.0) (-e 60.0)
  ```
    - `-t|--type` refers to the SLAM method type
    - `-f|--file` refers to the recorded rosbag that contains the gt/estimated poses
    - `-s|--st` refers to the **start time** for evaluation
    - `-e|--et` refers to the **end time** for evaluation