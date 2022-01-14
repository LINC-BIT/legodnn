'''
feature_learner + anomaly_learner

iterative self-training

average all output scores of models in each iteration
'''

import torch
from torch import nn

from cv_task.anomaly_detection.methods.ornet.anomaly_score_learner.anomaly_learner import AnomalyLearner
from cv_task.anomaly_detection.methods.ornet.anomaly_score_learner.feature_learner import FeatureLearner


class AnomalyScoreLearner(nn.Module):
    def __init__(self, feature_learner, m):
        super(AnomalyScoreLearner, self).__init__()
        
        self.feature_learner = feature_learner
        self.anomaly_learner = AnomalyLearner(m)
        
    def forward(self, x):
        x = self.feature_learner(x)
        x = self.anomaly_learner(x)
        return x
