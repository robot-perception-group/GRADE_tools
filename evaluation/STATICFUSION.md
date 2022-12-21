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

> The official instructions for MRPT library can be found [here](https://docs.mrpt.org/reference/latest/download-mrpt.html#downloadmrpt).

#### Install Other Dependencies

```bash
sudo apt install cmake freeglut3-dev libglew-dev libopencv-dev libopenni2-dev git
```

### 2. Installation

- Modify the **Camera Parameters** at **Line 124** of `StaticFusion.h` as following **before installation**:
  ```cpp
  StaticFusion(unsigned int width = 640 / 2, \
               unsigned int height = 480 / 2, \
               float fx =  325.910278 / 2, \
               float fy = 399.310333 / 2, \
               float cx = 320 / 2., \
               float cy = 240 / 2.);
  ```
- Build
  ```bash
  git clone git@github.com:christian-rauch/staticfusion.git
  cd staticfusion
  git checkout updates # Switch to this updated branch
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
- The expected format of the images:

    - Color images - 8 bit in PNG. Resolution: VGA
    - Depth images - 16 bit monochrome in PNG, scaled by 1000. Resolution: VGA

- `rgbd_assoc.txt` contain a list of items of the form as shown below, which can be downloaded from [here](https://github.com/Kyle-Xu001/Synthetic-Robotic-Data-Generation/blob/main/config/rgbd_assoc.txt) to associate corresponding color and depth images.
  ```
  timestamp1 /rgb/rgb_id.png timestamp2 /depth/depth_id.png
  ```
- Run: `./StaticFusion-ImageSeqAssoc PATH_TO_DATASET`
    - Run `sudo ldconfig` if error occurs when loading shared libraries: libpango_windowing.so
    - Click **SAVE** button after completing entire experiment to generate **estimated pose** `sf-mesh.txt` and **pointcloud** `st-mesh.ply` results in `/build` folder.

### 4.Evaluation

- Evaluate Estimated Result using Absolute Trajecotry Error (ATE) and Trajecotry Plot
  ```bash
  ./evaluate.sh -t staticfusion -f sf-mesh.txt (-s 0.0) (-e 60.0)
  ```
    - `-t|--type` refers to the SLAM method type
    - `-f|--file` refers to the estimated result from Tartan VO method
    - `-s|--st` refers to the **start time** for evaluation
    - `-e|--et` refers to the **end time** for evaluation
