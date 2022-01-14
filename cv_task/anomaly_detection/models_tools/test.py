import torch
# from cv_task.anomaly_detection.methods.ganomaly.lib.model import Ganomaly
# from cv_task.anomaly_detection.methods.ganomaly.options import Options
from torch.utils.data import Dataset, DataLoader
import copy

def test_ganomaly_model(netg, test_dataloader: DataLoader, dataset_name, device='cuda'):
    from cv_task.anomaly_detection.models.ganomaly import ganomaly_coil100_whole, ganomaly_emnist_whole
    batch_size = test_dataloader.batch_size
    if dataset_name=='coil100':
        model = ganomaly_coil100_whole()
    elif dataset_name=='emnist':
        model = ganomaly_emnist_whole()
    else:
         raise NotImplementedError
    model.opt.batchsize=batch_size
    model.netg = copy.deepcopy(netg)
    dataloader = dict(test=test_dataloader)
    test_res = model.test_model(dataloader)
    return test_res[0][0]

def test_gpnd_model(gpnd, train_loader: DataLoader, test_loader: DataLoader, dataset_name='caltech256', device='cuda'):
    assert dataset_name == 'caltech256'
    image_channels = 3
    image_size = 32
    latent_size = 128
    batch_size = 64
    from cv_task.anomaly_detection.methods.gpnd.test_AAE import test
    (auc, _, _), _ = test(gpnd.g, gpnd.e, train_loader, test_loader, image_channels, image_size, latent_size, batch_size)
    return auc

def test_ornet_model(model, test_loader: DataLoader, all_video_frames_label):
    from cv_task.anomaly_detection.methods.ornet.util.train_ofa import test
    auc, _, _ = test(model, test_loader, all_video_frames_label)
    return auc