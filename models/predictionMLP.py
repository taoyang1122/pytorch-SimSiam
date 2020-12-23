import torch.nn as nn


class PredictionMLP(nn.Module):
    def __init__(self, d=2048, hd=512):
        super(PredictionMLP, self).__init__()

        self.l1 = nn.Linear(d, hd)
        self.bn1 = nn.BatchNorm1d(hd)
        self.l2 = nn.Linear(hd, d)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.l1(x)))
        x = self.l2(x)
        return x
