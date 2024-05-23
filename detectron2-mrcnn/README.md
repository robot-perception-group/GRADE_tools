For detectron2:
1. Put the files in detectron2_train inside detectron2/tools
2. Edit train_coco.py folders and evaluations as desired.
3. Run train_coco.py

To evaluate the COCO model only on the person class:
1. copy detectron2/evaluation/evaluator.py
2. load COCO val2017 dataset and modify `train_coco.py` validation part with the following
```python
register_coco_instances("coco_val_custom", {}, "/home/ebonetto/detectron2/tools/COCO/annotations/instances_val2017.json",
                        "/home/ebonetto/detectron2/tools/COCO/val2017")
cfg = custom_config(80, "temp_filename", "coco_val_custom", "coco_val_custom", weights_path)
cfg.MODEL.WEIGHTS = './model_final_01ca85.pkl'
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("coco_val_custom", output_dir=os.path.join(cfg.OUTPUT_DIR, "val_full"))
val_loader = build_detection_test_loader(cfg, "coco_val_custom")
val_full = inference_on_dataset(predictor.model, val_loader, evaluator, custom_filter=True)
```
