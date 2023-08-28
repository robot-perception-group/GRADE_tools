## TRAINING DATASET GENERATION

This document will help you automatically generate the image datasets with **instance masks** and **bbox labels** from GRADE dataset.

In general you can use this tool as follows:
1. Edit the `config.yaml` file to customize your dataset. You can decide which data you want, the main path of your original generated data, the output folders, which viewports to use, and the image sizes
2. \[Optional\] Edit the `generate.py` and/or the `instance.py` files to customize the generation process. You can decide which object classes you want to detect, the mapping relations between input data and output dataset, and the ignored image IDs. Specifically, you can edit the extensions of the images, visualize the results, enable filtering etc.
3. Run `./generate.sh` to generate the dataset.
4. Run `python shuffle_full.py` to shuffle the dataset. Before doing that you want to edit the first few lines to account for the input/output folders as well adding any skeletal or other kind of information. This code takes care of merging the labels folders, shuffling, splitting, and creating the annotations in the COCO format.

For the GRADE indoor dataset we used the `detect_occlusion` function to remove images with possible occlusion or darkness.

### Experiment Structure
Running `generate.sh` will generate the following structure for each experiment:

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
- 
- `*_wrong_labels.txt` contains the object labels that cannot be mapped into NYU40 using `mapping.pkl`, which can be used to update the `mapping.pkl`.
- `*_mapping.txt` contains the mapping relations between input original data and output dataset. 
    > For example, `EXPERIMENT_ID/1500.png  object/1.png` refers that `1.png` in `/object` folder is generated from `1500.png` in `/EXPERIMENT_ID` folder.
- `*_ignored_ids.txt` contains the image IDs that are identified as **bad frame** with possible occlusion or darkness. These images will not be processed or added to the output dataset.
- `/object` includes the images and labels which contain the desired objects. The generated masks and labels will only involve detection of desired objects and do not involve other objects.
- `/non_object` includes the images and labels which **do not** contain the desired objects (also called background images). The generated masks and labels are empty.
    > Desired object classes can be defined in `object_classes` in `config.yaml`.

Running `python shuffle_full.py` will generate the following structure:
```
OUTPUT_FOLDER/
    ├── train/
    │   ├── images/
    │   ├── masks/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   ├── masks/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   ├── masks/
    │   └── labels/
    └── various mapping.txt
```


### Dataset Generation

The dataset can be either containing blurred images generated using our augmentation tools or the original images. The blurred images are generated using the `preprocessing/blur` tool. The original images are generated using the `preprocessing/extract_data` tool. 

Input data structrue should be as follows:
```
DATA_FOLDER/
    ├── EXP1/
        └── VIEWPORT/
                ├─── rgb/           # full size RGB image (1920X1080)
                ├─── depthLinear/   # full size depth npy file (1920X1080)
                ├─── instance/      # instance npy files
                ├─── keypoints/     # keypoints npy files
                └─── bbox_2d_tight/ # bbox npy files
    ├── EXP2/
    └── .../
```

Please specify the params in `config.yaml` to customize your dataset.

If you want to generate the dataset WITHOUT blurred images:
- Turn `noise` as `False` since it will generate the blur dataset
- Choose `bbox` and `instance` based on your desired output
- Choose specific classes in  `object_classes` to filter other classes in output
- Specify the `main_path`, `output_path`, `viewport_name` and `output_image_size`

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