#!/bin/bash
python main_simsiam.py \
--aug-plus \
--cos \
-a resnet50 \
--mult 0.5 \
-p 100 \
--lr 0.05 \
--batch-size 256 \
--epochs 100 \
--dist-url 'tcp://localhost:10001' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
/data/jeff-Dataset/ImageNet/raw-data/