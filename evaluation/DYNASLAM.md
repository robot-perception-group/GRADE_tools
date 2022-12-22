## DynaSLAM

### 1. Prerequisties

- Ubuntu 20.04
- C++ 11
- gcc/g++ 4.8 **[Recommended, it is an old repo. We provide info on how to build and install it locally such that you can keep multiple installations.]**
- Eigen 3
- [Pangolin-0.5](https://github.com/robot-perception-group/Pangolin)
- OpenCV 2.4.11 or 2.4.13 **[Recommended]**

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
  sudo apt install g++-4.8 gcc-4.8
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 10
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 10
  ```

- Change gcc and g++ version to 4.8
  ```bash
  sudo update-alternatives --config gcc
  sudo update-alternatives --config g++
  ```

- ###Remember to revert the source list file
- ###Remember to revert the alternatives when you finished compiling

#### Installing Pangolin-0.5 (with **gcc/g++ 4.8**)

```bash
sudo apt install libglew-dev cmake libpython2.7-dev
# clone modified pangolin repo
git clone git@github.com:robot-perception-group/Pangolin.git
cd Pangolin
mkdir build && cd build
cmake .. # to install locally [-D CMAKE_INSTALL_PREFIX=../install]
make -j && sudo make install # if local you do not need sudo
```

#### Installing OpenCV-2.4.11/2.4.13

- Install Successfully with **gcc/g++ 4.8**
  ```bash
  wget https://github.com/opencv/opencv/archive/refs/tags/2.4.11.zip
  unzip 2.4.11.zip
  rm 2.4.11.zip
  cd opencv-2.4.11 # or OpenCV-2.4.13
  mkdir build && cd build
  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../install -D WITH_CUDA=OFF -D WITH_FFMPEG=OFF -D ENABLE_PRECOMPILED_HEADERS=OFF -D BUILD_OPENEXR=ON ..
  make -j && sudo make install
  ```

### 2.Installation

As a suggestion, you should use a virtualenv. Instructions will not include that.

- Compile DynaSLAM with **gcc/g++ 4.8**

  ```bash
  sudo apt install libboost-all-dev
  git clone https://github.com/BertaBescos/DynaSLAM.git
  cd DynaSLAM
  chmod +x build.sh
  ```

  Apply the following edits:
  
  1. Delete following lines from `CMakeLists.txt` **Line 150 to Line 152**
  ```
  add_executable(mono_carla Examples/Monocular/mono_carla.cc)
  target_link_libraries(mono_carla ${PROJECT_NAME})
  ```
  2. Check Line 85 of `DynaSLAM/CMakeLists.txt` and make sure **numpy** is installed inside python 2.7

  3. Link the correct OpenCV version. Change Line 36 of `DynaSLAM/CMakeLists.txt` and Line 27 of `Thirdparty/DBoW2/CMakeLists.txt` as:

  ```cmake
  find_package(OpenCV 2.4.11 EXACT REQUIRED PATHS YOUR_OPENCV_PATH)
  ```

- Install Required Libaries for MASK-RCNN

  ```bash
  sudo apt install python-tk
  # Using CUDA 9.0 CUDNN 7.0
  # In case you miss them you can follow 
  # https://developer.nvidia.com/cuda-90-download-archive
  # and https://developer.nvidia.com/rdp/cudnn-archive v7.0.5 works
  # export PATH=/usr/local/cuda-9.0/bin:$PATH
  # export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
  pip2 install tensorflow-gpu==1.5.0 keras==2.1.6 scikit-image pycocotools
  # Without CUDA
  # pip2 install tensorflow==1.5.0 keras==2.1.6 scikit-image pycocotools
  ```

- Download pretrained model `mask_rcnn_coco.h5` from [here](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) and place it in the folder `DynaSLAM/src/python`

- Check Mask-RCNN work properly by running `python2 src/python/Check.py`
  > If everything is alright, it will print `Mask R-CNN is correctly working`

### 3. Run
- Prepare the data:
  Either extract the RGB-D from the rosbags or from the `npy` files. To do that, refer to [this]() step.
  TODO add link + structure data
  
  
- Execute the following command. Change `PATH_TO_DATA_FOLDER` to the sequence folder. `PATH_TO_MASKS` and `PATH_TO_OUTPUT` are optional parameters.

  ```
  ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt /path_to_GRADE-eval/evaluation/config/DynaSLAM-mpi.yaml \
                            PATH_TO_DATA_FOLDER /path_to_GRADE-eval/evaluation/config/rgbd_assoc.txt (PATH_TO_MASKS) (PATH_TO_OUTPUT)
  ```
  
   #### Notes

  - If `PATH_TO_MASKS` and `PATH_TO_OUTPUT` are **not** provided, only the geometrical approach is used to detect dynamic objects.
  - If `PATH_TO_MASKS` is provided, Mask R-CNN is used to segment the potential dynamic content of every frame. These masks are saved in the provided folder.
  - If `PATH_TO_MASKS` is set to `no_save`, the masks are used but not saved. 
  - If it finds the Mask R-CNN computed dynamic masks in `PATH_TO_MASKS`, it uses them but does not compute them again.
  - If `PATH_TO_OUTPUT` is provided, the inpainted frames are computed and saved in `PATH_TO_OUTPUT`.
  
- The organisational structure of the dataset should be:
  ```
  /dataset/rgb/   # folder containing all color images
  /dataset/depth/ # folder containing all depth images
  /dataset/masks/ # empty folder to save masks
  /dataset/results/ # empty folder to save output
  /dataset/rgbd_assoc.txt # association file
  ```
- The expected format of the images:
    - Color images - 8 bit in PNG. Resolution: VGA
    - Depth images - 16 bit monochrome in PNG, scaled by 1000. Resolution: VGA

### 4. Evaluation

- Estimated result will be saved in `CameraTrajectory.txt` at `DynaSLAM` folder
- Evaluate Estimated Result using Absolute Trajecotry Error (ATE) and Trajecotry Plot
  ```bash
  ./evaluate.sh -t dynaslam -f ESTIMATED_RESULT_TXT -g GROUNTRUTH_BAG (-o OUTPUTDIR) (-s 0.0) (-e 60.0)
  ```
    - `-t|--type` refers to the SLAM method type
    - `-f|--file` refers to the estimated result from DynaSLAM method
    - `-s|--st` refers to the **start time** for evaluation
    - `-e|--et` refers to the **end time** for evaluation
    - `-o|--od` refers to the output directory
    - `-g|--gb` refers to the bags from which you want to extract the groundtruth

### 5. Run on your own data
You can run on your own data. 
The main thing you need to take care is the association that you can obtain with
`python evaluation/src/tool/associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt`

Change the yaml file according to your camera parameters.