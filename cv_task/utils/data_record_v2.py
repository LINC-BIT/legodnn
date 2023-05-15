#!D:/Code/python
# -*- coding: utf-8 -*-
# @Time : 2021/5/15 0015 15:05
# @Author : xgf
# @File : data_record_xgf.py
# @Software : PyCharm

import argparse
import math
import time
# import pickle
# import pickle5 as pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
# fix random seed
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import csv
import time
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import importlib

class DataRecord:
    state_dict = {}
    def __init__(self, model_save_file):
        # self._time = str(time.strftime("%Y-%m-%d/%H-%M-%S", time.localtime()))
        # self._log_path = os.path.join(self.root_path, dataset_name, self.method_name, self._time)
        self.model_save_file = model_save_file
        self._log_path = os.path.dirname(model_save_file)
        if not os.path.exists(self._log_path):
            os.makedirs(self._log_path)
            print('log_path:', self._log_path)

    def record_opt(self, opt):
        self._opt_path = self._log_path + '/opt.txt'
        f = open(self._opt_path, 'w')
        f.write(str(opt))
        f.close()

    def GPND_loss_write_csv(self, loss_list, name):
        name = '/loss_' + name + '.csv'
        loss_path = self._log_path + name
        with open(loss_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'D', 'G', 'ZD', 'GE', 'E'])

    def state_dict_update(self, key_value_list):
        for key, value in key_value_list:
            if key not in self.state_dict:
                self.state_dict[key] = []
            self.state_dict[key].append(value)
        np.save(self._log_path + '/state_dict.npy', self.state_dict)

    def save_model(self, model_name, checkpoint):
        self._model_path = os.path.join(self._log_path, model_name)
        torch.save(checkpoint, self._model_path)

    def record_report(self, report_str):
        self._report_path = self._log_path + '/report.txt'
        f = open(self._report_path, 'a')
        f.writelines(report_str + '\n')
        f.close()
