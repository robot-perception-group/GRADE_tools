import torch, detectron2
import sys, os, distutils.core

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine.hooks import HookBase, BestCheckpointer
from detectron2.config import CfgNode
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.file_io import PathManager
from LossEvalHook import *

cfg = get_cfg()

import json
from detectron2.data.datasets import register_coco_instances


def custom_config(num_classes, outdir, train, val, weights=''):
    cfg = get_cfg()

    # get configuration from model_zoo
    cfg.merge_from_file(model_zoo.get_config_file("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml"))
    cfg.DATASETS.TRAIN = (train,)
    cfg.DATASETS.TEST = (val,)
    cfg.DATALOADER.NUM_WORKERS = 8

    cfg.MODEL.WEIGHTS = weights

    img_ratio = 4 / (8 * 2)
    cfg.SOLVER.IMS_PER_BATCH = 16 * img_ratio
    cfg.SOLVER.BASE_LR = 0.02 * img_ratio

    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.TEST.IMS_PER_BATCH = 1
    cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes

    cfg.OUTPUT_DIR = outdir

    cfg.SOLVER.BEST_CHECKPOINTER_SEGM = CfgNode({"ENABLED": True})
    cfg.SOLVER.BEST_CHECKPOINTER_SEGM.METRIC = "segm/AP50"
    cfg.SOLVER.BEST_CHECKPOINTER_SEGM.MODE = "max"

    cfg.SOLVER.BEST_CHECKPOINTER_BBOX = CfgNode({"ENABLED": True})
    cfg.SOLVER.BEST_CHECKPOINTER_BBOX.METRIC = "bbox/AP50"
    cfg.SOLVER.BEST_CHECKPOINTER_BBOX.MODE = "max"

    cfg.TEST.EVAL_PERIOD = 2000

    cfg.SOLVER.MAX_ITER = 270000 # --- [2 images] 120 --- [4 images] 90
    cfg.SOLVER.STEPS = (210000, 250000) # --- 80,108 --- 60K, 80K

    return cfg


class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_subfolder="./eval"):
        coco_evaluator = COCOEvaluator(dataset_name, output_dir=os.path.join(cfg.OUTPUT_DIR, output_subfolder))
        evaluator_list = [coco_evaluator]
        return evaluator_list

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))

        if self.cfg.SOLVER.BEST_CHECKPOINTER_SEGM and comm.is_main_process():
            hooks.append(BestCheckpointer(
                self.cfg.TEST.EVAL_PERIOD,
                self.checkpointer,
                self.cfg.SOLVER.BEST_CHECKPOINTER_SEGM.METRIC,
                mode=self.cfg.SOLVER.BEST_CHECKPOINTER_SEGM.MODE,
                file_prefix="best_segm",
            ))

        if self.cfg.SOLVER.BEST_CHECKPOINTER_BBOX and comm.is_main_process():
            hooks.append(BestCheckpointer(
                self.cfg.TEST.EVAL_PERIOD,
                self.checkpointer,
                self.cfg.SOLVER.BEST_CHECKPOINTER_BBOX.METRIC,
                mode=self.cfg.SOLVER.BEST_CHECKPOINTER_BBOX.MODE,
                file_prefix="best_bbox",
            ))

        # swap the order of PeriodicWriter and ValidationLoss
        # code hangs with no GPUs > 1 if this line is removed
        hooks = hooks[:-2] + hooks[-2:][::-1]
        return hooks

    def build_writers(self):
        """
        Overwrites the default writers to contain our custom tensorboard writer

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        PathManager.mkdirs(self.cfg.OUTPUT_DIR)
        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            CustomTensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]


if __name__ == '__main__':
    train_name = "coco_red_train"
    val_name = "coco_red_val"
    weights_path = ""

    register_coco_instances(train_name, {}, "/home/ebonetto/train_humans.json",
                            "/home/ebonetto/coco/train/images")
    register_coco_instances(val_name, {}, "/home/ebonetto/coco/val_humans.json",
                            "/home/ebonetto/coco/val2017")
    register_coco_instances("coco_val_full_filtered", {}, "/home/ebonetto/coco/val_humans.json",
                            "/home/ebonetto/coco/val2017")
    register_coco_instances("tum_val", {}, "/home/ebonetto/tum_valid/annotations/instances_val2017.json",
                            "/home/ebonetto/tum_valid/val2017")

    MetadataCatalog.get(train_name).thing_classes = ["person"]
    MetadataCatalog.get(val_name).thing_classes = ["person"]
    MetadataCatalog.get("coco_val_full_filtered").thing_classes = ["person"]
    MetadataCatalog.get("tum_val").thing_classes = ["person"]

    cfg = custom_config(1, "fullcoco", train_name, val_name, weights_path)
    # import ipdb; ipdb.set_trace()
    #
    #trainer = CocoTrainer(cfg)
    #trainer.resume_or_load(resume=(weights_path != ''))
    #trainer.train()

    from detectron2.data import build_detection_test_loader

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    import ipdb; ipdb.set_trace()

    evaluator = COCOEvaluator("coco_val_full_filtered", output_dir=os.path.join(cfg.OUTPUT_DIR, "val_full"))
    val_loader = build_detection_test_loader(cfg, "coco_val_full_filtered")
    val_full = inference_on_dataset(predictor.model, val_loader, evaluator)

    evaluator = COCOEvaluator("tum_val", output_dir=os.path.join(cfg.OUTPUT_DIR, "val_tum"))
    val_loader = build_detection_test_loader(cfg, "tum_val")
    val_tum = inference_on_dataset(predictor.model, val_loader, evaluator)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "thr_0.05"), exist_ok=True)
    np.save(os.path.join(cfg.OUTPUT_DIR, "thr_0.05", "val_full_final.npy"), val_full)
    np.save(os.path.join(cfg.OUTPUT_DIR, "thr_0.05", "val_tum_final.npy"), val_tum)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_segm.pth")
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("coco_val_full_filtered", output_dir=os.path.join(cfg.OUTPUT_DIR, "val_full"))
    val_loader = build_detection_test_loader(cfg, "coco_val_full_filtered")
    val_full = inference_on_dataset(predictor.model, val_loader, evaluator)

    evaluator = COCOEvaluator("tum_val", output_dir=os.path.join(cfg.OUTPUT_DIR, "val_tum"))
    val_loader = build_detection_test_loader(cfg, "tum_val")
    val_tum = inference_on_dataset(predictor.model, val_loader, evaluator)
    np.save(os.path.join(cfg.OUTPUT_DIR, "thr_0.05", "val_full_segm.npy"), val_full)
    np.save(os.path.join(cfg.OUTPUT_DIR, "thr_0.05", "val_tum_segm.npy"), val_tum)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_bbox.pth")
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("coco_val_full_filtered", output_dir=os.path.join(cfg.OUTPUT_DIR, "val_full"))
    val_loader = build_detection_test_loader(cfg, "coco_val_full_filtered")
    val_full = inference_on_dataset(predictor.model, val_loader, evaluator)

    evaluator = COCOEvaluator("tum_val", output_dir=os.path.join(cfg.OUTPUT_DIR, "val_tum"))
    val_loader = build_detection_test_loader(cfg, "tum_val")
    val_tum = inference_on_dataset(predictor.model, val_loader, evaluator)
    np.save(os.path.join(cfg.OUTPUT_DIR, "thr_0.05", "val_full_bbox.npy"), val_full)
    np.save(os.path.join(cfg.OUTPUT_DIR, "thr_0.05", "val_tum_bbox.npy"), val_tum)

############################################################################################################
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "thr_0.7"), exist_ok=True)
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("coco_val_full_filtered", output_dir=os.path.join(cfg.OUTPUT_DIR, "val_full"))
    val_loader = build_detection_test_loader(cfg, "coco_val_full_filtered")
    val_full = inference_on_dataset(predictor.model, val_loader, evaluator)

    evaluator = COCOEvaluator("tum_val", output_dir=os.path.join(cfg.OUTPUT_DIR, "val_tum"))
    val_loader = build_detection_test_loader(cfg, "tum_val")
    val_tum = inference_on_dataset(predictor.model, val_loader, evaluator)
    np.save(os.path.join(cfg.OUTPUT_DIR, "thr_0.7", "val_full_final.npy"), val_full)
    np.save(os.path.join(cfg.OUTPUT_DIR, "thr_0.7", "val_tum_final.npy"), val_tum)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_segm.pth")
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("coco_val_full_filtered", output_dir=os.path.join(cfg.OUTPUT_DIR, "val_full"))
    val_loader = build_detection_test_loader(cfg, "coco_val_full_filtered")
    val_full = inference_on_dataset(predictor.model, val_loader, evaluator)

    evaluator = COCOEvaluator("tum_val", output_dir=os.path.join(cfg.OUTPUT_DIR, "val_tum"))
    val_loader = build_detection_test_loader(cfg, "tum_val")
    val_tum = inference_on_dataset(predictor.model, val_loader, evaluator)
    np.save(os.path.join(cfg.OUTPUT_DIR, "thr_0.7", "val_full_segm.npy"), val_full)
    np.save(os.path.join(cfg.OUTPUT_DIR, "thr_0.7", "val_tum_segm.npy"), val_tum)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_bbox.pth")
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("coco_val_full_filtered", output_dir=os.path.join(cfg.OUTPUT_DIR, "val_full"))
    val_loader = build_detection_test_loader(cfg, "coco_val_full_filtered")
    val_full = inference_on_dataset(predictor.model, val_loader, evaluator)

    evaluator = COCOEvaluator("tum_val", output_dir=os.path.join(cfg.OUTPUT_DIR, "val_tum"))
    val_loader = build_detection_test_loader(cfg, "tum_val")
    val_tum = inference_on_dataset(predictor.model, val_loader, evaluator)
    np.save(os.path.join(cfg.OUTPUT_DIR, "thr_0.7", "val_full_bbox.npy"), val_full)
    np.save(os.path.join(cfg.OUTPUT_DIR, "thr_0.7", "val_tum_bbox.npy"), val_tum)
