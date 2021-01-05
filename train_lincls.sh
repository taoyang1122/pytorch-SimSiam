#!/bin/bash
python main_lincls.py \
-a resnet50 \
--lr 0.02 \
--cos \
--batch-size 2048 \
-p 100 \
--pretrained logs/R50e100_nofc/checkpoint_0099.pth.tar \
--dist-url 'tcp://localhost:10001' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
/home/ubuntu/yang/data/ImageNet/ILSVRC/Data/CLS-LOC