import torch
from cv_task.anomaly_detection.methods.ganomaly.lib.model import Ganomaly
from cv_task.anomaly_detection.methods.ganomaly.options import Options
from torchvision.models import resnet
# LOAD_MODE = ['netg', 'whole']

def ganomaly_coil100_netg(pretrained=True, device='cuda'):
    if pretrained:
        netg = torch.load('/data/gxy/legodnn-auto-on-cv-models/cv_task_model/anomaly_detection/ganomaly_coil100/ganomaly_coil100.best').to(device)
    netg.eval()
    return netg

def ganomaly_coil100_whole(pretrained=False, learning_rate=0.0002, epoch_num=60, image_channel=3, image_size=32, lantent_size=100, batch_size=128, device='cuda', model_save_path=None):
    opt = Options().parse()
    opt.nc = image_channel
    opt.isize = image_size
    opt.nz = lantent_size
    opt.batchsize = batch_size
    opt.device=device
    opt.niter = epoch_num
    opt.lr = learning_rate
    opt.model_save_path = model_save_path
    model = Ganomaly(opt)
    model.reset_optimizer()
    if pretrained:
        model.netg = torch.load('/data/gxy/legodnn-auto-on-cv-models/cv_task_model/anomaly_detection/ganomaly_coil100/ganomaly_coil100.best')
    # model.eval()
    return model

def ganomaly_emnist_netg(pretrained=True, device='cuda'):
    if pretrained:
        netg = torch.load('/data/gxy/legodnn-auto-on-cv-models/cv_task_model/anomaly_detection/ganomaly_emnist/ganomaly_emnist.best').to(device)
    netg.eval()
    return netg

def ganomaly_emnist_whole(pretrained=False, learning_rate=0.00002, epoch_num=30, image_channel=1, image_size=32, lantent_size=256, batch_size=128, device='cuda', model_save_path=None):
    opt = Options().parse()
    opt.nc = image_channel
    opt.isize = image_size
    opt.nz = lantent_size
    opt.batchsize = batch_size
    opt.device=device
    opt.niter = epoch_num
    opt.lr = learning_rate
    opt.model_save_path = model_save_path
    model = Ganomaly(opt)
    model.reset_optimizer()
    if pretrained:
        model.netg = torch.load('/data/gxy/legodnn-auto-on-cv-models/cv_task_model/anomaly_detection/ganomaly_emnist/ganomaly_emnist.best')
    # model.eval()
    return model
# def 