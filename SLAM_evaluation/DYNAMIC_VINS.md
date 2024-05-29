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

  Modify `sophus/so2.cpp` as

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

We will use a modified fork. The edit has been made to avoid crashing of the program when tracking was lost.
`yolo_ros` is also already included in this forked repo.

```bash
cd {your workspace}/src
# clone modified fork
git clone git@github.com:robot-perception-group/Dynamic-VINS.git

# install python dependencies
sudo apt install ros-noetic-ros-numpy python3-pip
python3 -m pip install --upgrade pip
```

We suggest to use a virtual env. You can do that either with `conda` or `python-venv`.

```bash
sudo apt install python3.8-venv
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate
# install pytorch with CUDA based on your device (or conda install)
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install -r /{your-catkin}/src/Dynamic-VINS/yolo_ros/requirements.txt # you may need to remove an opecv-python from the file
pip3 install python3-empy
```

#### Note that the build process need to be made while sourcing the venv/conda environment.
```bash
# build
cd {your_catkin}
catkin_make
source devel/setup.bash
```

- Change Python Path at **Line 1** of `/{your-catkin}/src/Dynamic-VINS/yolo_ros/src/demo_service_server.py` to your venv/conda environment.
- Make sure that packages `camera_model` and `vins_estimator` in `Dynamic-VINS` are using the OpenCV from ROS (version 4.2) instead of other OpenCV built from source, otherwise it will fail when loading `/EstimatorNodelet`.

### 3. Run

#### Be sure to source the venv/conda environment.

When you launch the estimator you need to wait to see `service mode : yolo_service` appearing in the terminal.
That indicates that yolo has been loaded and is ready.

Note that since we are playing bags, one should ensure that `use_sim_time` parameter is set to True.

**Since the estimation process depends on the speed of your PC, slowing down the `rosbag play` (`-r 0.5`) may improve results.**

Remember to check the config yaml file (`config/tum_rgbd/...`) for the maximum distance of the depth map.

You can either launch the various parts separatedly or, as done for testing, use the `grade_pytorch.launch` and the `tum_pytorch.launch` files with the rosbag play, vins_estimator and rosbag record running together. Similar launch files have been made for the VIO version. The yaml configuration files have been adapted for both the 3.5m and the 5m depth ranges. Please take care of using the correct one based on your data. See the automated scripts for bulk testing (either in this repo or in the forked repository).

- Visual Inertial Odometry (GRADE):
  ```bash
  roslaunch vins_estimator openloris_vio_pytorch_mpi.launch \
                  rgb:="/my_robot_0/camera_link/0/rgb/image_raw" \
                  depth:="/my_robot_0/camera_link/0/depth/image_raw" \
                  imu:="/my_robot_0/imu_body"
  roslaunch vins_estimator vins_rviz.launch
  rosbag play /dataset/*.bag --clock -r 0.2
  ```
- Visual Odometry (GRADE Dataset):
  ```bash
  roslaunch vins_estimator tum_rgbd_pytorch_mpi.launch \
  rgb:="/my_robot_0/camera_link/0/rgb/image_raw" \
  depth:="/my_robot_0/camera_link/0/depth/image_raw"
  roslaunch vins_estimator vins_rviz.launch
  rosbag play /dataset/*.bag --clock  -r 0.2
  ```
- Visual Odometry (Default TUM RGBD-Dataset). For this, please follow the DynaVINS instructions in the [repo](github.com:robot-perception-group/Dynamic-VINS.git) and change the intrinsic parameters (i.e. 325.9 -> 460 in .cpp and .h files). OR checkout the `tum_params` branch:
  ```bash
  roslaunch vins_estimator tum_rgbd_pytorch.launch
  roslaunch vins_estimator vins_rviz.launch
  rosbag play /dataset/*.bag --clock -r 0.2
  ```
- To save Estimated Result during experiments:
  ```bash
  rosbag record /my_robot_0/camera/pose \
                /my_robot_0/odom \
                /vins_estimator/camera_pose \
                /vins_estimator/init_map_time -O result.bag
  ```
#### Known issues:
- `Service call failed: service [/yolo_service] responded with an error: error processing request: 'Upsample' object has no attribute 'recompute_scale_factor'` look [here](https://github.com/openai/DALL-E/issues/54) how to solve this.

### 4. Evaluation

- Evaluate Estimated Result for **GRADE** using Absolute Trajectory Error (ATE), Relative Pose Error (RPE) and Trajecotry Plot
  ```bash
  ./evaluate.sh -t dynavins -f result.bag (-s 0.0) (-e 60.0)
  ```
    - `-t|--type` refers to the SLAM method type
    - `-f|--file` refers to the recorded rosbag that contains the gt/estimated poses
    - `-s|--st` refers to the **start time** for evaluation
    - `-e|--et` refers to the **end time** for evaluation

- Evaluate Estimated Result for **TUM-RGBD** using Absolute Trajectory Error (ATE), Relative Pose Error (RPE) and Trajecotry Plot
  ```bash
  ./evaluate.sh -t dynavins_tum -f result.bag
  ```
    - `-t|--type` refers to the SLAM method type
    - `-f|--file` refers to the recorded rosbag that contains the gt/estimated poses
  
### 5. Run your own data

Change the parameters in `{catkin_ws}/src/Dynamic-VINS/vins_estimator/launch/openloris/openloris_vio_pytorch_mpi.launch` or `{catkin_ws}/src/Dynamic-VINS/vins_estimator/launch/tum_rgbd/tum_rgbd_pytorch_mpi.launch`

Especially `imu/image/depth topic, estimate_td, image_width, image_height, projection-parameters, extrinsicTranslation/Rotation`.

For more information follow the README in the forked/official repo. 

Note that you may want to change `output_dynavins` in `output_pose.py` to account for your extrinsics and your topics.
