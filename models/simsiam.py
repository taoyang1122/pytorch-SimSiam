import torch
import torch.nn as nn
import math
# from models.resnet import resnet50
from torchvision.models import resnet50
from .predictionMLP import PredictionMLP


class SimSiam(nn.Module):

    def __init__(self, backbone='resnet50', d=2048):
        super(SimSiam, self).__init__()

        if backbone == 'resnet50':
            net = resnet50()
        else:
            raise NotImplementedError('Backbone model not implemented.')

        num_ftrs = net.fc.in_features
        self.features = nn.Sequential(*list(net.children())[:-1])
        # num_ftrs = net.fc.out_features
        # self.features = net

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, d)
        self.bn1 = nn.BatchNorm1d(d)
        self.l2 = nn.Linear(d, d)
        self.bn2 = nn.BatchNorm1d(d)
        self.l3 = nn.Linear(d, d)
        self.bn3 = nn.BatchNorm1d(d)
        self.relu = nn.ReLU()
        # self.l1 = nn.Sequential(
        #     nn.Linear(num_ftrs, d),
        #     nn.BatchNorm1d(d),
        #     nn.ReLU()
        # )
        # self.l2 = nn.Sequential(
        #
        # )

        # prediction MLP
        self.prediction = PredictionMLP()

        self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # projection
        x = self.relu(self.bn1(self.l1(x)))
        x = self.relu(self.bn2(self.l2(x)))
        z = self.bn3(self.l3(x))
        # prediction
        p = self.prediction(z)
        return z, p

    def reset_parameters(self):
        # reset conv initialization to default uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                stdv = 1. / math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)