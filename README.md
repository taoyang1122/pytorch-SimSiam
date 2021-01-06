# Exploring simple siamese representation learning

This is a PyTorch re-implementation of the [SimSiam paper](https://arxiv.org/abs/2011.10566). The implementation is based on the codes of [MOCO](https://github.com/facebookresearch/moco).

## Unsupervised pre-training
To run unsupervised pre-training on ImageNet,
```
sh train_simsiam.sh
```
This is to reproduce the results of ```batch-size=256, epoch=100``` setting. Please modify the path to your ImageNet data folder.

## Linear evaluation
To run linear evaluation,
```
sh train_lincls.sh
```
The linear evaluation is done using LARS optimizer with a batch size of 4096. ```Lr=0.32 (0.02*4096/256)```, ```epoch=90```.

|SimSiam|batchsize|100 ep|
|-------|---------|------|
|Reported|256|67.7%|
|Reproduced|256|66.0%|

## Issues
Currently, I wasn't able to reproduce the results. I listed some possible issues below. Any discussions are welcome.

1. The paper didn't say the applying probability of the Blur Augmentation. I am following the setting of MOCO which is p=0.5.
