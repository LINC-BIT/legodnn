'''
two-layer FC network
'''

import torch
from torch import nn


class AnomalyLearner(nn.Module):
    def __init__(self, M):
        super(AnomalyLearner, self).__init__()
        
        self.linear1 = nn.Linear(M, 100)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(100, 1)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x
