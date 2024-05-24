1. Put the two `py` files in detectron2_train inside detectron2/tools
2. Edit `train_coco.py` folders and evaluations as desired.
3. Run `train_coco.py`

`LossEvalHook` is there to be able to perform validation step (which is super-slow) every few epochs and be able to save the best model. It also takes care of populating the tensorboard correctly.

In `train_coco.py` you can modify the `custom_config` function to load a different basic configuration and other parameters such as the number of epochs, the learning rate steps etc.

You then need to define your train/val sets names, the path of the weights that you want to load eventually.
Each dataset has been created following the COCO convention, so a JSON file + image folder. 
You will need one training, one validation and eventually some additional test datasets. 
Then the configuration is loaded and expanded according to `custom_config` and the model trained.
For the evaluation part, we use the COCO definitions as well, running COCOEvaluator with the various datasets and saving the different best models. 

To evaluate the baseline COCO model only on the person class you can either use the full `val2017` dataset, and modify the code to show per-class AP50 (per-class AP is reported by default), or consider using just one class. We provide in the published data the COCO validation set already filtered with only the humans. You may want to use that json file or the full json with all cats.

In the paper, we evaluate only on the people, as it is more consistent to what we train. Thus, we use only the 2693 images with people on them and those labels.
The procedure is:

1. Download [model_final_f10217](https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl) (if you use `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml` config) or [model_final_01ca85.pkl](https://dl.fbaipublicfiles.com/detectron2/Misc/scratch_mask_rcnn_R_50_FPN_3x_gn/138602908/model_final_01ca85.pkl) (if you use `Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml` config)
2. copy `detectron2/evaluation/evaluator.py` to the corresponding file. We have an additional method that discards all the non-persons detections _prior_ final processing and evaluation. This is useful if your json file does not have all the 80 categories.
3. Run the following:

```python
from detectron2.data import build_detection_test_loader

weights_path = 'model_final_01ca85.pkl' # if using Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml config
# weights_path = 'model_final_f10217.pkl' # if using COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml -> in this case please change also the base config path on `custom_config` to match

register_coco_instances("coco_val_ppl", {}, "./COCO/val_people.json", "./COCO/valid/images")

cfg = custom_config(80, "coco_baseline", "coco_val_ppl", "coco_val_ppl", weights_path)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
# warnings like "Skip loading parameter 'roi_heads.box_predictor.cls_score.weight' to the model due to incompatible shapes: (81, 1024) in the checkpoint but (2, 1024) in the model! You might want to double check if this is expected." are _expected_ if you use only 1 class
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("coco_val_ppl", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "coco_val_ppl")
inference_on_dataset(predictor.model, val_loader, evaluator) # use custom_filter=True if the json has only one category defined
```