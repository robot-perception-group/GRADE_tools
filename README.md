# GRADE PROCESSING & EVALUATION

### This repository is part of the [GRADE](https://eliabntt.github.io/GRADE-RR/home) project.

Many of the tools (especially the noise scripts) can be used for different projects and different scopes.

This repository contains all the files necessary to process the data generated by our [GRADE framework](https://eliabntt.github.io/GRADE-RR/home).

Find more information about the GRADE by looking at the [paper](https://arxiv.org/abs/2303.04466), the [video](https://www.youtube.com/watch?v=cmywCSD-9TU&t=1s), or the [project website](https://eliabntt.github.io/GRADE-RR/home).

## Quick Guide
- To obtain **rosbags** with reordered timestamps or noise, please refer to [bag processing](preprocessing/PREPROCESSING.md#L74) part
- To obtain (noisy) **rgb/depth images (640X480)**, etc., please refer to [data extraction](preprocessing/PREPROCESSING.md#L88) part
- To obtain **noisy rgb/depth images (1920X1080)**, etc., please refer to [file processing](preprocessing/PREPROCESSING.md#L88) part
- To obtain **training dataset** with bbox labels and instance masks, please refer to [dataset generation](dataset_generation/GENERATION.md) part
- To obtain instructions for **SLAM** deployments, please refer to [SLAM evaluation](SLAM_evaluation/EVAL.md) part
___
## (Pre)Processing  -  Add Noise, Reorder Timestamps...
The first step you need to do is the pre-processing of the experiment data.
You can find all the necessary information [here](preprocessing/PREPROCESSING.md).
This will guide you on adding noise to the data, limitting the depth, generating rgb/depth images, and much more.
___
## SLAM Evaluation
Then you can run evaluation scripts. You can find all the information [here](SLAM_evaluation/EVAL.md).
Those are the settings used in the evaluations that you can find in the paper.
___
## Semantic Mapping
The semantic information output by Issac differs from the NYUv2-40 mapping.
Therefore, we made a script (which is appliable to bboxes, instance, semantic info) to get the correct mappings [here]().
___
## Identify Bad Frames
Possible occlusions, subjects that are too nearby the camera, camera that look only toward outside the environment and many others.
If you want to check these things, there is a snippet that might help you out [here](https://github.com/robot-perception-group/GRADE-eval/blob/main/mapping_and_visualization/convert_classes.py#L60)
___
## Additional Scripts
- IMU visualization to check eventual spikes and smoothness of the IMU data, as well as the added noise 
  [here](https://github.com/robot-perception-group/GRADE-eval/blob/main/additional_scripts/imu_visualize.py)
  For this script you need to specify `[data,imu,imu_noisy]_dir` in lines 6:8 of the python code.
  
- Timestamp verification script that will output plots and stats to check whether the original bags have sorted timestamps as expected. Note that this is an error of  `isaac_sim` itself. You can find the script [here](https://github.com/robot-perception-group/GRADE-eval/blob/main/additional_scripts/timestamp_verification.py).

- Ap36K converter notebook to convert the APT-36K dataset into usable filtered sets.

- convert COCO to npy, COCO to YOLO, and npy to COCO scripts to convert the dataset into the desired format.

- filter coco dataset to remove classes that are not desired from the original COCO annotations and be able to reduce the original data/create annotations which have only the desired class. e.g. `python filter_coco_json.py -i ./coco_ds/annotations/instances_val2017.json -o ./val_people.json -c person`

- Visualize COCO json to be able to quickly load and see annotated in a notebook the COCO annotations generated.

- Get Keypoints show how the data generated with GRADE can be processed to get the keypoints. 

_______

### LICENSING
For licensing information, please refer to the main repository located [here](https://github.com/eliabntt/GRADE-RR/).
__________

### CITATION
If you find this work useful, please cite our work based on [this](https://github.com/eliabntt/GRADE-RR#citation) information

_____

## Acknowledgments
We would like to thank [Chenghao Xu](https://github.com/Kyle-Xu001) for the support, testing and development of many building pieces of this code.

[rotors_simulator](https://github.com/ethz-asl/rotors_simulator) for the piece of code that we integrated to add noise to the IMU readings.

Please refer also to the official implementations of 
- [RTABMAP](https://github.com/introlab/rtabmap)
- [Dynamic-VINS](https://github.com/HITSZ-NRSL/Dynamic-VINS)
- [YOLO_ROS](https://github.com/hirokiyokoyama/yolo_ros)
- [DynaSLAM](https://github.com/BertaBescos/DynaSLAM)
- [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)
- [STATICFUSION](https://github.com/raluca-scona/staticfusion)
- [TartanVO](https://github.com/castacks/tartanvo)
- [VDO_SLAM](https://github.com/halajun/VDO_SLAM)

for any update or issue that you have regarding their code. We may be able to help, but please check them out :) 
