## TRAINING DATASET GENERATION

This document will help you automatically generate the image datasets with **instance masks** and **bbox labels** from GRADE dataset.

The generation process will remove images with possible occlusion or darkness. Then it will generate the corresponding instance masks and bboxes labels, which ***only contain the object classes we want*** and can be directly trained by MASK-RCNN and YOLOv5.


### Datasets Structure
```
DATASET_FOLDER/
    ├── *_wrong_labels.txt     # objects cannot be mapped into NYU40
    ├── *_mapping.txt          # mapping relations
    ├── *_ignored_ids.txt      # ignored image ID
    ├── object/                # dataset with desired objects
         ├─── images/
         ├─── masks/
         └─── labels/          # bbox labels
    └── non_object/            # dataset without desired objects
         ├─── images/
         ├─── masks/
         └─── labels/          # bbox labels
```
- `*_wrong_labels.txt` contains the object labels that cannot be mapped into NYU40 using `mapping.pkl`, which can be used to update the `mapping.pkl`.
- `*_mapping.txt` contains the mapping relations between input original data and output dataset. 
    > For example, `EXPERIMENT_ID/1500.png  object/1.png` refers that `1.png` in `/object` folder is generated from `1500.png` in `/EXPERIMENT_ID` folder.
- `*_ignored_ids.txt` contains the image IDs that are identified as **bad frame** with possible occlusion or darkness. These images will not be processed or added to the output dataset.
- `/object` includes the images and labels which contain the desired objects. The generated masks and labels will only involve detection of desired objects and do not involve other objects.
- `/non_object` includes the images and labels which **do not** contain the desired objects (also called background images). The generated masks and labels are empty.
    > Desired object classes can be defined in `object_classes` in `config.yaml`.


### Dataset Generation
#### **Generating Default Dataset (non-blur/non-noisy)**
Input data structrue should as follows:
```
DATA_FOLDER/
    ├── EXP1/
        └── VIEWPORT/
                ├─── rgb/           # full size RGB image (1920X1080)
                ├─── depthLinear/   # full size depth npy file (1920X1080)
                ├─── instance/      # instance npy files
                └─── bbox_2d_tight/ # bbox npy files
    ├── EXP2/
    └── .../
```
Please specify the params in `config.yaml` to customize your dataset:
- Turn `noise` as `False` since it will generate the blur dataset
- Choose `bbox` and `instance` based on your desired output
- Choose specific classes in  `object_classes` to filter other classes in output
- Specify the `main_path`, `output_path`, `viewport_name` and `output_image_size`
#### **Generating Blur Image Dataset**
You have two options to generate blur image dataset with ground truth labels. Both of them require **blur parameter files** which you can generate in `preprocessing/bag_process`. The bbox labels will be generated using instance information in blur dataset. Thus, please always provide instance data in this genereation.
- Option 1: Extract blur images(640X480) from rosbag, then formulate the blur datasets
    - Turn `blur/save/enable` in `preprocessing/config/bag_process.yaml` as `True`. Then start `bag_process` in [here](../preprocessing/PREPROCESSING.md) to save blur params in `/noisy_bags/data`. 
    - Extract images from rosbag using `preprocessing/extract_data` in [here](../preprocessing/PREPROCESSING.md).
    - Specify the `blur_param_path` [here](src/generate.py#L110) and `blur_img_path` [here](src/generate.py#L116)
    - Turn `noise` as `True` and `blur_img_exist` as `True`
- Option 2: Generate blur images from full-size RGB images(1920X1080) during formulation
    - Turn `blur/save/enable` in `preprocessing/config/bag_process.yaml` as `True`. Then start `bag_process` in [here](../preprocessing/PREPROCESSING.md) to save blur params in `/noisy_bags/data`. 
    - Specify the `blur_param_path` [here](src/generate.py#L110)
    - Turn `noise` as `True` and `blur_img_exist` as `False`

> NOTE 1: Two options may generate different blur images since the same homography matrix results in different blur effects on images with different sizes.

> NOTE 2: If you want to reproduce the result in our paper, please set `blur/config/exposure_time` as 0.02 in `bag_process.yaml` and apply **option 1**. You can find more information in branch `mask_blur` [here](https://github.com/robot-perception-group/GRADE_tools/tree/mask_blur).

#### **RUN**
```
cd GRADE_tools/dataset_generation
./generate.sh
```
> Please check input data folders [here](src/generate.py#L98) before running.