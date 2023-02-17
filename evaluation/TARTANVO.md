## TartanVO
### 1. Prerequisites

- Ubuntu 20.04
- CUDA (Test with Version **9.2**, follow [this](https://developer.nvidia.com/cuda-92-download-archive))
- Python == 3.7.0
- opencv-python == 4.2.0.32
- pytorch == 1.4.0
- cupy-cuda92 == 9.6.0

Use the following command to install all dependencies (as always we suggest a venv):

```bash
pip3 install matplotlib scipy torch==1.4.0 opencv-python==4.2.0.32 cupy-cuda92==9.6.0 numpy==1.23.0
```

### 2. Pretrained Model

```bash
git clone git@github.com:robot-perception-group/tartanvo.git
cd tartanvo
mkdir models
wget https://cmu.box.com/shared/static/t1a5u4x6dxohl89104dyrsiz42mvq2sz.pkl -O models/tartanvo_1914.pkl
```

### 3. Run

- The organisational structure of the dataset should be:
  ```
  /dataset/rgb/   # folder containing all color images
  /dataset/gt_pose_tartan.txt
  ```

- `gt_pose_tartan.txt` contain a list of items of the form as shown below:
  ```
  pos_x pos_y pos_z quat_x quat_y quat_z quat_w
  ```
  > The pose refers to the **ground truth camera pose** with reference to the first frame, which means **the first ground truth pose data should be `[0. 0. 0. 0. 0. 0. 1.]`**

- Generate `gt_pose_tartan.txt` by running:
    - Generate Default Pose Ground Truth File `gt_pose.txt`
      ```bash
      python3 GRADE-eval/evaluation/src/tool/output_pose.py --type groundtruth \
                              --path BAG_SEQUENCE_FOLDER[reindex_bags_folder] \
                              --topic /my_robot_0/camera/pose
      ```
    - Transform `gt_pose.txt` to generate `gt_pose_tartan.txt`
      ```bash
      python3 GRADE-eval/evaluation/src/tool/output_pose.py --type tartan_gt --path gt_pose.txt
      ```
- Run:
  ```bash
  python vo_trajectory_from_folder.py --grade --model-name tartanvo_1914.pkl \
                                      --test-dir /dataset/rgb \
                                      --pose-file /dataset/gt_pose_tartan.txt \
                                      --batch-size 1 \
                                      --worker-num 1
  ```

### 4. Evaluation

- The system will automatically evaluate the absolute trajectory error (ATE) after completing the experiments. The estimated poses and ATE plot results will be saved in `./results` folder.
- (Optional) Evaluate Estimated Result using Absolute Trajectory Error (ATE), Relative Pose Error (RPE) and Trajecotry Plot

  `ESTIMATED_RESULT_TXT` is by default `./results/grade_tartanvo_1914.txt`.
  
   In general it goes by `./results/[dataset_name]_[model_name].txt`
  
  ```bash
  ./evaluate.sh -t tartan -f ESTIMATED_RESULT_TXT -g GROUNDTRUTH_BAG_FOLDER (-o OUTPUTDIR) (-s 0.0) (-e 60.0)
  ```
    - `-t|--type` refers to the SLAM method type
    - `-f|--file` refers to the estimated result from Tartan VO method
    - `-s|--st` refers to the **start time** for evaluation
    - `-e|--et` refers to the **end time** for evaluation
    - `-o|--od` refers to the output dir
    - `-g|--gb` refers to the folder of grountruth bags from which we extract `gt_pose.txt`
### 5. Your own data

Add your own intrinsic on the `Dataset/utils.py` file.
Add your own dataset on `vo_trajectory_from_folder.py` `get_args()` function and at the beginning of the `__main__`.