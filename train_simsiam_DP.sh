#!/bin/bash
python main_simsiam_DP.py \
--aug-plus \
--cos \
-a resnet50 \
--lr 0.1 \
-p 100 \
--epochs 100 \
--batch-size 512 \
--dist-url 'tcp://localhost:10001' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
/data/ImageNet/CLS-LOC/