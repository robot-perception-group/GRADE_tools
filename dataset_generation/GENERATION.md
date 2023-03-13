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

```
cd GRADE_tools/dataset_generation
./generate.sh
```

Please save the blur params when generating the noisy bags in [here](../preprocessing/config/bag_process.yaml#L72)
