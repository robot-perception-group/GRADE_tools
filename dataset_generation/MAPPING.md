## TRAINING DATASET GENERATION

This document will help you automatically generate the image datasets with **instance masks** and **bbox labels** from GRADE dataset.


Please save the blur params when generating the noisy bags in [here](../preprocessing/config/bag_process.yaml#L72)

Masks and image should be the same size

Mask will be resize to 640,480 and then resize to 960 720