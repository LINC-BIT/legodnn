'''
pretrained ResNet-50
'''

import torch
from torch import nn
from cv_task.anomaly_detection.methods.ornet.pretrained_model.model import ft_net


class FeatureLearner(nn.Module):
    def __init__(self):
        super(FeatureLearner, self).__init__()
        
        self.ft_net = ft_net(751)
        
    def forward(self, x):
        x = self.ft_net(x)
        return x
