#!/bin/bash
python main_simsiam_DP.py \
--aug-plus \
--cos \
-a resnet50 \
--lr 0.05 \
--batch-size 64 \
--dist-url 'tcp://localhost:10001' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
/home/ubuntu/yang/data/ImageNet/ILSVRC/Data/CLS-LOC