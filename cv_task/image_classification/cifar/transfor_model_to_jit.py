from math import gamma
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
# from models import *
# from utils import progress_bar
from xgf_model.cifar10.models import *
from xgf_model.cifar10.utils import progress_bar, format_time
from xgf_model.utils.data_record_xgf import DataRecordXgf

        # raw_model = SENet18() # load pretrained weight!
        # raw_model = torch.nn.DataParallel(raw_model)
        # raw_model.load_state_dict(torch.load('../xgf_model/save_model/cifar10/SENet18/2021-06-17/18-27-37/SENet18.pth')['net'])
        # raw_model = raw_model.module

# net = SENet18()
# net = torch.nn.DataParallel(net)
# net.load_state_dict(torch.load('/data/zql/legodnn-on-var-models/xgf_model/save_model/cifar10/SENet18/2021-06-17/18-27-37/SENet18.pth')['net'])
# net = net.module
# # print(net)

# net.eval()
# example = torch.rand(1, 3, 32, 32)
# traced_script_module = torch.jit.trace(net, example, check_trace=False)
# torch.jit.save(net, "/data/zql/legodnn-on-var-models/xgf_model/save_model/cifar10/SENet18/2021-06-17/18-27-37/SENet18.jit")

pretained_model = SENet18()
pretained_model = torch.nn.DataParallel(pretained_model)
pretained_model.load_state_dict(torch.load('/data/zql/legodnn-on-var-models/xgf_model/save_model/cifar10/SENet18/2021-06-17/18-27-37/SENet18.pth')['net'])    # 网络+权重
pretained_model = pretained_model.module
# 载入为单gpu模型
# gpu_model       = pretained_model.module  # GPU-version
# 载入为cpu模型
# model           = SENet18()
# pretained_dict  = pretained_model.module.state_dict()
# model.load_state_dict(pretained_dict)  # CPU-version

pretained_model.eval()
example = torch.rand(1, 3, 32, 32)
traced_script_module = torch.jit.trace(pretained_model, example, check_trace=False)
torch.jit.save(traced_script_module, "/data/zql/legodnn-on-var-models/xgf_model/save_model/cifar10/SENet18/2021-06-17/18-27-37/SENet18-cpu.jit")