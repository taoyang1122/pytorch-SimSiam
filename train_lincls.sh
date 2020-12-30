#!/bin/bash
python main_lincls.py \
-a resnet50 \
--lr 30.0 \
--batch-size 256 \
--pretrained log/R50e100/checkpoint_0099.pth.tar \
--dist-url 'tcp://localhost:10001' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
/home/ubuntu/yang/data/ImageNet/ILSVRC/Data/CLS-LOC