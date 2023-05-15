
import os 
import sys
sys.setrecursionlimit(100000)
import copy
import torch
from tqdm import tqdm

from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from legodnn.block_detection.model_topology_extraction import topology_extraction
from legodnn.block_detection.block_extraction_11_28 import LegoDNNBlock
from legodnn.utils.dl.common.pruning import l1_prune_model, l1_prune_model_by_dummy_input
from legodnn.utils.dl.common.env import set_random_seed
from legodnn.utils.dl.common.model import get_model_flops_and_params
from legodnn.utils.common.file import ensure_dir, experiments_model_file_path, remove_dir
from legodnn.utils.common.data_record import CSVDataRecord
from legodnn.utils.dl.common.model import save_model, ModelSaveMethod,  get_model_flops_and_params, get_model_size, get_module, set_module

from cv_task.anomaly_detection.models.ganomaly import ganomaly_coil100_whole


def train_ganomaly_coil100_whole(cv_task, dataset_name, model_name, method, epoch_num, train_loader, test_loader, device='cuda'):
    # model_save_path = '/data/gxy/legodnn-auto-on-cv-models/cv_task_model/anomaly_detection/ganomaly_coil100/ganomaly_coil100'
    model_save_file = os.path.join(os.path.dirname(experiments_model_file_path('./', cv_task, dataset_name, model_name, method, epoch_num, 0.0)), method + '_' + model_name)
    ensure_dir(model_save_file)
    model = ganomaly_coil100_whole(model_save_path=model_save_file, device=device)
  
    dataloader = dict(train=train_loader, test=test_loader)
    model.train_model(dataloader)


def train_gpnd_caltech256_model(train_epoch_num, cv_task, dataset_name, model_name, method, epoch_num, train_loader, test_loader, device='cuda'):
    from cv_task.anomaly_detection.methods.gpnd.net import NewGenerator, NewEncoder, NewGPND
    from cv_task.anomaly_detection.methods.gpnd.train_AAE import train
    from .test import test_gpnd_model
    image_channel = 3
    image_size = 32
    latent_size = 128
    # epoch_num = 300
    learning_rate = 0.001
    batch_size = 64
    
    # model_save_path = '/data/gxy/legodnn-auto-on-cv-models/cv_task_model/anomaly_detection/gpnd_caltech256/gpnd_caltech256'
    model_save_file = os.path.join(os.path.dirname(experiments_model_file_path('./', cv_task, dataset_name, model_name, method, epoch_num, 0.0)), method + '_' + model_name)
    ensure_dir(model_save_file)
    device = 'cuda'
    
    G = NewGenerator(latent_size, channels=image_channel).to(device)
    G.weight_init(mean=0, std=0.02)
    E = NewEncoder(latent_size, channels=image_channel).to(device)
    E.weight_init(mean=0, std=0.02)
    gpnd = train(G, E, model_save_file, train_loader, image_channel, image_size, latent_size, train_epoch_num, batch_size, learning_rate)

    auc = test_gpnd_model(gpnd, train_loader, test_loader, dataset_name, device)
    print("GPND 模型auc为: {}".format(auc))


def train_ornet_model(train_epoch_num, cv_task, dataset_name, model_name, method, epoch_num, learning_rate=0.001, milestones=[30], device='cuda'):
    from torchvision.models.resnet import resnet18
    from cv_task.anomaly_detection.methods.ornet.anomaly_score_learner import AnomalyScoreLearner
    from cv_task.anomaly_detection.methods.ornet.util.train import train_us_net, test, public_test_loader, public_all_video_frames_label, train_ornet_xgf

    # learning_rate = 0.001
    model_save_file = os.path.join(os.path.dirname(experiments_model_file_path('./', cv_task, dataset_name, model_name, method, epoch_num, 0.0)), method + '_' + model_name)
    ensure_dir(model_save_file)

    backbone = resnet18(pretrained=True).to(device)
    model = AnomalyScoreLearner(backbone, 1000).to(device)
    print(model)
    exit()
    train_ornet_xgf(model, '/data/zql/mcw/model/ornet/ucsd-ped1-raw-data.pth', 
        '/data/zql/mcw/model/ornet/ucsd-ped1-AN-tensor.pth', model_save_file, train_epoch_num, lr=learning_rate, milestones=milestones, init_epoch_num=3)