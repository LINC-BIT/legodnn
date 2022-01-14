

import copy
import shutil

from torchvision.models.alexnet import alexnet
from torchvision.models.resnet import resnet18, resnet50
from torchvision.models.vgg import vgg16
import sys

sys.path.insert(0, '../../')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from functools import reduce
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import MinMaxScaler

from cv_task.anomaly_detection.methods.ornet.util.datasets import UCSDDataset, get_extracted_features
from cv_task.anomaly_detection.methods.ornet.util.train import train, test
from cv_task.anomaly_detection.methods.ornet.init_anomaly_detection import split_A_N, split_A_N_by_score
from cv_task.anomaly_detection.methods.ornet.anomaly_score_learner import AnomalyScoreLearner
from cv_task.anomaly_detection.methods.ornet.anomaly_score_learner.feature_learner import FeatureLearner

# import sys
# sys.path.insert(0, '/data/zql/zedl')
# from zedl.dl.common.env import set_random_seed
# set_random_seed(0)


