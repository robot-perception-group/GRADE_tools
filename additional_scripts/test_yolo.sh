#!/bin/bash
source ~/yolov5/bin/activate
cd /media/ebonetto/WindowsData/GRADE-nets/yolov5/

python val.py --data data/zebra_r1.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC/weights/best.pt > r1_d1_sc.txt 2>&1
python val.py --data data/zebra_r1.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC-1920/weights/best.pt --imgsz 1920 > r1_d1_sc-1920.txt 2>&1
python val.py --data data/zebra_r1.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/RP/weights/best.pt > r1_d1_rp.txt 2>&1
python val.py --data data/zebra_r1.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/RP-1920/weights/best.pt --imgsz 1920 > r1_d1_rp-19200.txt 2>&1
python val.py --data data/zebra_r1.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC+COCO/weights/best.pt  --imgsz 1920 > r1_d1_sc_coco.txt 2>&1
python val.py --data data/zebra_r1.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC+C+R3/weights/best.pt --imgsz 1920 > r1_d1_sc_c_r3.txt 2>&1
python val.py --data data/zebra_r1.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC+COCO+RP/weights/best.pt --imgsz 1920 > r1_d1_sc_coco_rp.txt 2>&1

python val.py --data data/zebra_r3.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC/weights/best.pt > r3_d1_sc.txt 2>&1
python val.py --data data/zebra_r3.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC-1920/weights/best.pt --imgsz 1920 > r3_d1_sc-1920.txt 2>&1
python val.py --data data/zebra_r3.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/RP/weights/best.pt > r3_d1_rp.txt 2>&1
python val.py --data data/zebra_r3.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/RP-1920/weights/best.pt --imgsz 1920 > r3_d1_rp-19200.txt 2>&1
python val.py --data data/zebra_r3.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC+COCO/weights/best.pt  --imgsz 1920 > r3_d1_sc_coco.txt 2>&1
python val.py --data data/zebra_r3.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC+C+R3/weights/best.pt --imgsz 1920 > r3_d1_sc_c_r3.txt 2>&1
python val.py --data data/zebra_r3.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC+COCO+RP/weights/best.pt --imgsz 1920 > r3_d1_sc_coco_rp.txt 2>&1

# change d1 to d2 in data/zebra_r1.yaml
sed -i 's/d1/d2/g' data/zebra_r1.yaml
python val.py --data data/zebra_r1.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC/weights/best.pt > r1_d2_sc.txt 2>&1
python val.py --data data/zebra_r1.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC-1920/weights/best.pt --imgsz 1920 > r1_d2_sc-1920.txt 2>&1
python val.py --data data/zebra_r1.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/RP/weights/best.pt > r1_d2_rp.txt 2>&1
python val.py --data data/zebra_r1.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/RP-1920/weights/best.pt --imgsz 1920 > r1_d2_rp-19200.txt 2>&1
python val.py --data data/zebra_r1.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC+COCO/weights/best.pt  --imgsz 1920 > r1_d2_sc_coco.txt 2>&1
python val.py --data data/zebra_r1.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC+C+R3/weights/best.pt --imgsz 1920 > r1_d2_sc_c_r3.txt 2>&1
python val.py --data data/zebra_r1.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC+COCO+RP/weights/best.pt --imgsz 1920 > r1_d2_sc_coco_rp.txt 2>&1

sed -i 's/d1/d2/g' data/zebra_r3.yaml
python val.py --data data/zebra_r3.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC/weights/best.pt > r3_d2_sc.txt 2>&1
python val.py --data data/zebra_r3.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC-1920/weights/best.pt --imgsz 1920 > r3_d2_sc-1920.txt 2>&1
python val.py --data data/zebra_r3.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/RP/weights/best.pt > r3_d2_rp.txt 2>&1
python val.py --data data/zebra_r3.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/RP-1920/weights/best.pt --imgsz 1920 > r3_d2_rp-19200.txt 2>&1
python val.py --data data/zebra_r3.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC+COCO/weights/best.pt  --imgsz 1920 > r3_d2_sc_coco.txt 2>&1
python val.py --data data/zebra_r3.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC+C+R3/weights/best.pt --imgsz 1920 > r3_d2_sc_c_r3.txt 2>&1
python val.py --data data/zebra_r3.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/SC+COCO+RP/weights/best.pt --imgsz 1920 > r3_d2_sc_coco_rp.txt 2>&1

sed -i 's/1: drone//g' data/zebra_r1.yaml
sed -i 's/1: drone//g' data/zebra_r3.yaml
python val.py --data data/zebra_r1.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/RP+C/weights/best.pt  --imgsz 1920 > r1_d2_rp_coco.txt 2>&1 # 1 class
python val.py --data data/zebra_r3.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/RP+C/weights/best.pt  --imgsz 1920 > r3_d2_rp_coco.txt 2>&1
sed -i 's/d2/d1/g' data/zebra_r1.yaml
sed -i 's/d2/d1/g' data/zebra_r3.yaml
python val.py --data data/zebra_r1.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/RP+C/weights/best.pt  --imgsz 1920 > r1_d1_rp_coco.txt 2>&1 # 1 class
python val.py --data data/zebra_r3.yaml --weights /ps/project/irotate/syn_zebras/yolo_models/RP+C/weights/best.pt  --imgsz 1920 > r3_d1_rp_coco.txt 2>&1
