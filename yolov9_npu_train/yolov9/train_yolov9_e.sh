#!/bin/bash

export ASCEND_LAUNCH_BLOCKING=1

python  train_dual.py \
--workers 8  \
--batch 16 --epochs 100 --img 512 --device 0  --min-items 0 --close-mosaic 15  --noval \
--data /home/ma-user/work/data.yaml \
--cfg /home/ma-user/work/yolov9/models/detect/yolov9-e.yaml \
--hyp /home/ma-user/work/yolov9/data/hyps/hyp.scratch-high.yaml  \
--name yolov9_e_


