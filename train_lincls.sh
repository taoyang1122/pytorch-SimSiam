#!/bin/bash
python main_lincls.py \
-a resnet50 \
--lr 0.32 \
--cos \
--epochs 90 \
--batch-size 4096 \
-p 100 \
--pretrained logs/R50e100_nofc/checkpoint_0099.pth.tar \
--dist-url 'tcp://localhost:10001' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
/data/ImageNet/CLS-LOC/