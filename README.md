# Exploring simple siamese representation learning

This is a PyTorch re-implementation of the [SimSiam paper](https://arxiv.org/abs/2011.10566) on ImageNet dataset. The results match that reported in the paper. The implementation is based on the codes of [MOCO](https://github.com/facebookresearch/moco).

## Unsupervised pre-training
To run unsupervised pre-training on ImageNet,
```
sh train_simsiam.sh
```
This is to do the unsupervised pre-training for 100 epochs. Please modify the path to your ImageNet data folder.

Note 1: I try to follow the setting in the paper, which is bs=512 and lr=0.1 on 8-GPU, but somehow I can not fit it. So I used the max batch_size that I can fit (432) while kept the lr unchaged (0.1).

Note 2: In pre-training, I didn't fix the lr of prediction MLP. According to the paper (Table. 1), fixing the lr of prediction MLP can give slightly improvements (67.7% -> 68.1%). You can try it if interested.

## Linear evaluation
To run linear evaluation,
```
sh train_lincls.sh
```
The linear evaluation is done using NVIDIA LARC optimizer by setting ```trus_coefficient=0.001``` and ```clip=False```. The batch size is 4096. 

Note: I first followed the setting in the paper, which is ```Lr=0.32 (0.02*4096/256)```. But I can only got a result of 66.0%. Then I increased the learning rate to ```Lr=1.6 (0.1*4096.256)``` and achieved the result of 67.8%. The results and models are given below.

|SimSiam|pretrained batchsize|lincls Lr|Top-1 Acc|
|-------|--------------------|---------|---------|
|Reported|512|0.32|67.7%|
|Reproduced|432 ([Model](https://drive.google.com/file/d/1cIkZ9krrjfBh1YAm5X-38N1XLkgRFPhP/view?usp=sharing))|1.6|67.8% ([Model](https://drive.google.com/file/d/1uk_U-I8hiQQiAi5S66fJFqRIoYAEYq7C/view?usp=sharing))|
|Reproduced|432|0.32|66.0%|

