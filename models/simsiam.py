import torch
import torch.nn as nn
from models.resnet import resnet50


class SimSiam(nn.Module):

    def __init__(self, backbone='resnet50', d=2048):
        super(SimSiam, self).__init__()

        if backbone == 'resnet50':
            net = resnet50()
        else:
            raise NotImplementedError('Backbone model not implemented.')

        num_ftrs = net.fc.in_features
        self.features = nn.Sequential(*list(net.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, d)
        self.bn1 = nn.BatchNorm1d(d)
        self.l2 = nn.Linear(d, d)
        self.bn2 = nn.BatchNorm1d(d)
        self.l3 = nn.Linear(d, d)
        self.bn3 = nn.BatchNorm1d(d)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn1(self.l1(x)))
        x = self.relu(self.bn2(self.l2(x)))
        x = self.bn3(self.l2(x))
        return x
