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
The linear evaluation exactly follows the training setting in MOCO, but it is not the setting used in the paper. In the paper, the author did the linear evaluation with ```batch-size=4096``` which is infeasible to me. According to the paper (Appendix. A), the MOCO setting will give ~1% lower accuracy. My reproduced results and models are given below.

[[Pre-training model]](https://drive.google.com/file/d/1kH4Ge6u-UKEiJ-Nii3X7AI22-FkAUO-p/view?usp=sharing) | [[Linear evaluation model]](https://drive.google.com/file/d/1xbwUceR9WX0uBQWCWnv7HPvdDmuJpbNL/view?usp=sharing)

|SimSiam|batchsize|100 ep|
|-------|---------|------|
|Reported|256|~67.0%|
|Reproduced|256|64.8%|

## Issues
Currently, I wasn't able to reproduce the results. I listed some possible issues below. Any discussions are welcome.

1. During pre-training, I use the R50 backbone with fc layer (out_dim=1000). Currently, I am re-doing the pre-training without the fc layer (output from the pooling layer to the following MLP projection).

2. I used the default R50 initialization in pytorch models, which is different from the paper. I modified the initialization according to the paper and am re-doing the pre-training.

3. The paper didn't say the applying probability of the Blur Augmentation. I am following the setting of MOCO which is p=0.5.
