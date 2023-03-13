## TRAINING DATASET GENERATION

This document will help you automatically generate the image datasets with **instance masks** and **bbox labels** from GRADE dataset.

The generation process will remove images with possible occlusion or darkness. Then it will generate the corresponding instance masks and bboxes labels, which ***only contain the object classes we want*** and can be directly trained by MASK-RCNN and YOLOv5.


### Datasets Structure
```
DATASET_FOLDER/
    ├── *_wrong_labels.txt     # objects cannot be mapped into NYU40
    ├── *_mapping.txt          # mapping relationships
    ├── *_ignored_ids.txt      # ignored image ID
    ├── object/                # dataset with desired objects
         ├─── images/
         ├─── masks/
         └─── labels/
    └── non_object/            # dataset without desired objects
         ├─── images/
         ├─── masks/
         └─── labels/
```
- `*_wrong_labels.txt` contains the object labels that cannot be mapped using `mapping.pkl`, which can be used to update the `mapping.pkl`
```bash
cd GRADE_tools/dataset_generation
./generate.sh
```

Please save the blur params when generating the noisy bags in [here](../preprocessing/config/bag_process.yaml#L72)

Masks and image should be the same size

Mask will be resize to 640,480 and then resize to 960 720