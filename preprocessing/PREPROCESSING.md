# Data Preprocess

This document will help you processing the data recorded during the experiments of the GRADE dataset.

There are many options.

The first choice you need to make is if you want to process the rosbags or the full size data (the saved `npy` files).
In the first case go to [ROSBAG Processing](#rosbag-processing), in the latter go to [File Processing](#file-processing).

In practice, given an input folder, the script can:
- select the correct section of the rosbags (after the `starting_experiment` signal)
- correct the topic times of the bags to account for possible wrong clock times due to `isaac_sim` known issues
- add noise to the IMU and depths
- limit the recorded depth to a maximum distance
- add motion blur noise and rolling shutter effect to the RGB data
- extract rgb and depth PNG images
- extract odom and imu in npy files
- ...

The `process_data.sh` will take care of:
- reindexing the bag (if there is a `.bag.active`)
- start the correct processing script (located in `src/[file,bag]_process/play_[file,bags].py`)
 

______
### CX-TODO: WRITE WHAT HAPPENS more or less (folders created etc) not super precise but at least a rough idea.
IIRC if noisy is true we'll have _noisy appended and possibly different folders.
Where is the MAIN output folder specified?
______

## 1. ROSBAG Processing

```bash
./process_data.sh -t bag -p [PATH_TO_YOUR_DATA]
```

- Customized parameters are defined in `config/bag_process.yaml`. Please refer to that for the complete set of parameters.
  - If message timestamps are overlapped, set `time_correction/enable` to `True` to generate reindex bags.
  - Set `noise` to `True` to generate **Noisy Bags**
  - Set `camera/pointcloud` to `True` to generate **Pointcloud** in Noisy Bags
  - Set `blur/enable` to `True` to generate **Blurry Image** in Noisy Bags
- Reindex Bags will be saved in `/reindex_bags` folder
- Noisy Bags will be saved in `/noisy_bags` folder

## 2. Data extraction
- Extract Data from Rosbag

  ```bash
  ./process_data.sh -t extract -p [PATH_TO_YOUR_BAGS]
  ```

## 3. File Processing

  ```bash
  ./process_data.sh -t file -p [PATH_TO_YOUR_DATA]/Viewport
  ```

- Customized parameters are defined in `config/file_process.yaml`
  - #TODO this is a bit confusing, both why saving from bags is included in processing and there is little explanation on what is "extract data". Be a bit more wordy pls.
  - Set `extract_bag/enable` to `True` to extract data from rosbags (**Default reindex bags**)
  - Set `extract_bag/noisy` to `True` to also extract data from noisy rosbags
  - Set `extract_bag/save_images/enable` to `True` to save target rgb and depth PNG images (640X480)
  - Set `extract_bag/save_files/enable` to `True` to save target odom/imu data in \*npy files
  - Set `camera/enable` to `True` to generate **Noisy Depth files** in \*npy files (1920X1080)
  - Set `camera/output_img` to `True` to generate **Original and Noisy Depth Images** in \*.png files (1920X1080)
  - Set `blur/enable` to `True` to generate **Blurry RGB Images** in \*.png files (1920X1080)
  - Set `imu/enable` to `True` to generate **Noisy IMU Data** in \*.npy files

#### Example
```bash
./process_data.sh -t file -p ~/exp/Viewport0/
```