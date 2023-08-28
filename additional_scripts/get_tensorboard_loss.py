import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import sys

if __name__=="__main__":
    args = sys.argv[1:]

    if len(args) == 0 or len(args) > 1:
        print("please provide only one argument")
        sys.exit()

    log_dir = args[0]

    event_accumulator = EventAccumulator(log_dir)
    event_accumulator.Reload()
    val_rpn_class = event_accumulator.Scalars("epoch_val_rpn_class_loss")
    val_rpn_bbox_loss = event_accumulator.Scalars("epoch_val_rpn_bbox_loss")
    val_mrcnn_class_loss = event_accumulator.Scalars("epoch_val_mrcnn_class_loss")
    val_mrcnn_bbox_loss = event_accumulator.Scalars("epoch_val_mrcnn_bbox_loss")
    val_mrcnn_mask_loss = event_accumulator.Scalars("epoch_val_mrcnn_mask_loss")

    x = [x.step for x in val_rpn_class]
    y = []
    for rc, rp, c, b, m in zip(val_rpn_class, val_rpn_bbox_loss, val_mrcnn_class_loss, val_mrcnn_bbox_loss, val_mrcnn_mask_loss):
        y.append(rc.value+rp.value+c.value+b.value+m.value)
    print(f"The min val loss is {min(y)} considering all the losses at epoch {y.index(min(y))+1}(starting from 1)")

    x = [x.step for x in val_rpn_class]
    y = []
    for rc, rp, c, b, m in zip(val_rpn_class, val_rpn_bbox_loss, val_mrcnn_class_loss, val_mrcnn_bbox_loss, val_mrcnn_mask_loss):
        y.append(0*rc.value+0*rp.value+0*c.value+b.value+m.value)
    print(f"The min val loss is {min(y)} considering all the losses at epoch {y.index(min(y))+1}(starting from 1)")
