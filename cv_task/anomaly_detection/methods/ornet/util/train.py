from math import isnan
import sys

sys.path.insert(0, '../../')

import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from functools import reduce
from sklearn.metrics import auc, roc_curve, f1_score, precision_score, recall_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
import PIL
# from model.ornet.util.datasets import UCSDDataset, get_extracted_features
from cv_task.anomaly_detection.methods.ornet.init_anomaly_detection import split_A_N, split_A_N_by_score
from cv_task.anomaly_detection.methods.ornet.anomaly_score_learner import AnomalyScoreLearner
from cv_task.anomaly_detection.methods.ornet.util.mytensorsdataset import MyTensorsDataset

# sys.path.insert(0, '/data/zql/zedl')

from legodnn.utils.common.data_record import CSVDataRecord, write_json
from legodnn.utils.common.log import logger
from legodnn.utils.dl.common.model import save_model, ModelSaveMethod
from legodnn.utils.dl.common.env import set_random_seed
from legodnn.utils.common.file import ensure_dir

from baselines.nested_network.nestdnn_1230.nestdnn_open_api import zero_grads_nestdnn_layers
from baselines.nested_network.fn3_channel_open_api_1215.fn3_channel import set_fn3_channel_channels, export_active_sub_net
# sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/us_net')
# from open_api.slimmable_ops import USBatchNorm2d


# sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/cgnet')
# from cgnet_open_api.cg_layers import CGConv2d

# sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/once-for-all')
# from ofa_open_api.ofa_ornet import reset_running_BN

import numpy as np

    
def test(model, test_dataloader, all_video_frames_label, device='cuda'):
    model.eval()
    Y_pred = []
    
    for batch_index, (data, target) in enumerate(test_dataloader):
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            Y_pred += [output]
    
    Y_pred = torch.cat(Y_pred, dim=0)
    
    labels = all_video_frames_label.cpu()
    scores = Y_pred.cpu()
    
    threshold = 0.50
    scores[scores >= threshold] = 1
    scores[scores <  threshold] = 0
    
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    
    f1_res = f1_score(labels, scores)
    
    # return roc_auc, average_precision_score(labels, scores), f1_res
    return roc_auc

def reset_model_BN(model: torch.nn.Module, first_dataloader: DataLoader, device='cuda'):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
    model.train()
    for batch_index, (data, target) in enumerate(first_dataloader):
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            output = model(data)

from baselines.nested_network.usnet_open_api_1215.us_net import convert_model_to_us_net, set_us_net_width_mult, export_jit_us_sub_net, bn_calibration_init, set_FLAGS, export_us_sub_net, set_FLASGS_tmp

def train_ornet_xgf(model, all_video_raw_info_path, AN_init_info_path, model_save_path, epoch_num, batch_size=128, lr=0.001, milestones=[30], gamma=0.1, momentum=0.9, weight_decay=5e-4,device='cuda', init_epoch_num=0, **kwargs):
    logger.info('loading data...')
    ensure_dir(model_save_path)
    
    # data loading...
    all_video_tensor, all_video_frames_label = torch.load(all_video_raw_info_path)
    A_video_tensor, A_frames_index, N_video_tensor, N_frames_index = torch.load(AN_init_info_path)
    A_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[A_frames_index] == 1) / len(A_frames_index)
    N_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[N_frames_index] == 0) / len(N_frames_index)
    logger.info('cur train dataset | A correct rate: {:.6f}, N correct rate: {:.6f}'.format(A_correct_rate, N_correct_rate))
    
    X = torch.cat([A_video_tensor, N_video_tensor], dim=0).cpu()
    Y = torch.cat([
        torch.ones(A_video_tensor.size()[0]), 
        torch.zeros(N_video_tensor.size()[0])
    ]).unsqueeze(dim=1).cpu()
    anomaly_scores = np.zeros(all_video_tensor.size()[0])
    
    print(X.size(), Y.size(), all_video_tensor.size())
    
    # train loader
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    
    # test loader
    test_dataset = TensorDataset(all_video_tensor.cpu(), all_video_frames_label.cpu())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=4)
    
    bn_dataset = TensorDataset(all_video_tensor.cpu(), all_video_frames_label.cpu())
    bn_dataloader = DataLoader(bn_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4)
    
    # model = AnomalyScoreLearner().to(device)
    model = model.to(device)
    save_model(model, model_save_path, ModelSaveMethod.FULL, None)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    # epoch_num = 50
    criterion = torch.nn.L1Loss()
    
    model.train()
    best_roc_auc = 0
    
    logger.info('training...')
    import copy
    first_trainloader = copy.deepcopy(dataloader)
    method = kwargs.get('method', None)
    for epoch_index in range(epoch_num):
        train_loss = 0.
        
        # model.train()
        
        for batch_index, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            if method is not None:
                if method=='usnet':
                    # print('usnet')
                    width_mult_list = kwargs.get('width_mult_list', None)
                    sample_net_num = kwargs.get('sample_net_num', None)
                    assert width_mult_list is not None
                    assert sample_net_num is not None
                    min_width, max_width = min(width_mult_list), max(width_mult_list)
                    widths_train = []
                    for _ in range(sample_net_num - 2):
                        widths_train.append(
                            random.uniform(min_width, max_width))
                    widths_train = [max_width, min_width] + widths_train
                    optimizer.zero_grad()
                    for width in widths_train:
                        # print(width)
                        model.apply(lambda m: setattr(m, 'width_mult', width))
                        output = model(data)
                        loss = criterion(output, target)
                        train_loss += loss.item()
                        loss.backward()
                    optimizer.step()
                    model.apply(lambda m: setattr(m, 'width_mult', 1.0))
                elif method=='fn3':
                    fn3_all_layers = kwargs.get('fn3_all_layers', None)
                    fn3_disable_layers = kwargs.get('fn3_disable_layers', None)
                    min_sparsity = kwargs.get('min_sparsity', None)
                    
                    assert fn3_all_layers is not None
                    assert fn3_disable_layers is not None

                    fn3_channel_layers_name = [i[0] for i in fn3_all_layers]
                    fn3_channel_channels = [i[1] for i in fn3_all_layers]
                    # print(fn3_channel_channels)
                    # print(type(fn3_channel_channels[0]))
                    for i, c in enumerate(fn3_channel_channels):
                        if fn3_channel_layers_name[i] in fn3_disable_layers:
                            continue
                        fn3_channel_channels[i] = random.randint(int(min_sparsity*c), c)
                        
                    set_fn3_channel_channels(model, fn3_channel_channels)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    set_fn3_channel_channels(model, [i[1] for i in fn3_all_layers])
                elif method=='nestdnnv3':
                    zero_shape_info = kwargs.get('zero_shape_info', None)
                    assert zero_shape_info is not None
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    train_loss += loss.item()
                    loss.backward()
                    zero_grads_nestdnn_layers(model, zero_shape_info) # 清空前一个模型的梯度
                    optimizer.step()
                
                else:
                    raise NotImplementedError
            else:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            
        # scheduler.step()
        
        train_loss /= len(dataloader.dataset)
        # logger.info('epoch {}, train loss {:.6f}'.format(epoch_index, train_loss))
        
        if isnan(train_loss):
            logger.warn('loss exploded')
            return 0.5
        
        # reset_model_BN(model, first_trainloader)
        # reset_model_BN(model, bn_dataloader)
        # model.eval()
        # reset_running_BN(model, first_trainloader)
        if method is not None:
            if method=='usnet':
                print(1111)
                
                model.apply(bn_calibration_init)
                set_us_net_width_mult(model, 1.0)
                for batch_index, (data, target) in enumerate(dataloader):
                    data, target = data.to(device), target.to(device)
                    model(data)
            elif method=='fn3':
                pass
            elif method=='nestdnnv3':
                pass
            else:
                raise NotImplementedError    
        roc_auc = test(model, test_dataloader, all_video_frames_label, device)
        # logger.info('epoch {}, train loss {:.6f}, AUC: {:.6f}, best AUC: {:.6f}'.format(epoch_index, train_loss, roc_auc, best_roc_auc))

        # save_model(model, './raw-models/ucsd-ped1-model-{}.pth'.format(epoch_index), None, ModelSaveMethod.WEIGHT)
        # if roc_auc > best_roc_auc:
        #     best_roc_auc = roc_auc
        save_model(model, model_save_path, ModelSaveMethod.FULL, None)
        save_model(model, model_save_path + '.jit', ModelSaveMethod.JIT, (1, 3, 224, 224))
        save_model(model, model_save_path + '.weight', ModelSaveMethod.WEIGHT)
        logger.info('save best model in {}'.format(model_save_path))
        logger.info('epoch {}, train loss {:.6f}, AUC: {:.6f}, best AUC: {:.6f}'.format(epoch_index, train_loss, roc_auc, best_roc_auc))
            
        if epoch_index < init_epoch_num:
            continue
        
        # regenerate pseudo label
        model.eval()
        Y_pred = []
        for batch_index, (data, target) in enumerate(test_dataloader):
            with torch.no_grad():
                data = data.to(device)
                output = model(data)
                Y_pred += [output]
        Y_pred = torch.cat(Y_pred, dim=0).reshape(-1).cpu().detach().numpy()
        anomaly_scores += Y_pred
        
        A_video_tensor, A_frames_index, N_video_tensor, N_frames_index = split_A_N_by_score(all_video_tensor, anomaly_scores)
        A_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[A_frames_index] == 1) / len(A_frames_index)
        N_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[N_frames_index] == 0) / len(N_frames_index)
        logger.info('cur train dataset | A correct rate: {:.6f}, N correct rate: {:.6f}'.format(A_correct_rate, N_correct_rate))
        
        X = torch.cat([A_video_tensor, N_video_tensor], dim=0).cpu()
        Y = torch.cat([
            torch.ones(A_video_tensor.size()[0]), 
            torch.zeros(N_video_tensor.size()[0])
        ]).unsqueeze(dim=1).cpu()
        
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4)
        
        logger.info('updated A and N frames')
        
    logger.info('{} - {}'.format(model_save_path, best_roc_auc))
        
    return best_roc_auc

def train(model, all_video_raw_info_path, AN_init_info_path, model_save_path, epoch_num, batch_size=128, lr=0.001, device='cuda',
          init_epoch_num=0):
    logger.info('loading data...')
    ensure_dir(model_save_path)
    
    # data loading...
    all_video_tensor, all_video_frames_label = torch.load(all_video_raw_info_path)
    A_video_tensor, A_frames_index, N_video_tensor, N_frames_index = torch.load(AN_init_info_path)
    A_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[A_frames_index] == 1) / len(A_frames_index)
    N_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[N_frames_index] == 0) / len(N_frames_index)
    logger.info('cur train dataset | A correct rate: {:.6f}, N correct rate: {:.6f}'.format(A_correct_rate, N_correct_rate))
    
    X = torch.cat([A_video_tensor, N_video_tensor], dim=0).cpu()
    Y = torch.cat([
        torch.ones(A_video_tensor.size()[0]), 
        torch.zeros(N_video_tensor.size()[0])
    ]).unsqueeze(dim=1).cpu()
    anomaly_scores = np.zeros(all_video_tensor.size()[0])
    
    print(X.size(), Y.size(), all_video_tensor.size())
    
    # train loader
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=8)
    
    # test loader
    test_dataset = TensorDataset(all_video_tensor.cpu(), all_video_frames_label.cpu())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=8)
    
    bn_dataset = TensorDataset(all_video_tensor.cpu(), all_video_frames_label.cpu())
    bn_dataloader = DataLoader(bn_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=8)
    
    # model = AnomalyScoreLearner().to(device)
    model = model.to(device)
    save_model(model, model_save_path, ModelSaveMethod.FULL, None)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # epoch_num = 50
    criterion = torch.nn.L1Loss()
    
    model.train()
    best_roc_auc = 0
    
    logger.info('training...')
    import copy
    first_trainloader = copy.deepcopy(dataloader)
    for epoch_index in range(epoch_num):
        train_loss = 0.
        
        # model.train()
        
        for batch_index, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        train_loss /= len(dataloader.dataset)
        # logger.info('epoch {}, train loss {:.6f}'.format(epoch_index, train_loss))
        
        if isnan(train_loss):
            logger.warn('loss exploded')
            return 0.5
        
        # reset_model_BN(model, first_trainloader)
        # reset_model_BN(model, bn_dataloader)
        # model.eval()
        # reset_running_BN(model, first_trainloader)
        
        roc_auc, auprc, f1_res = test(model, test_dataloader, all_video_frames_label, device)
        # logger.info('epoch {}, train loss {:.6f}, AUC: {:.6f}, best AUC: {:.6f}'.format(epoch_index, train_loss, roc_auc, best_roc_auc))

        # save_model(model, './raw-models/ucsd-ped1-model-{}.pth'.format(epoch_index), None, ModelSaveMethod.WEIGHT)
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            save_model(model, model_save_path, ModelSaveMethod.FULL, None)
            save_model(model, model_save_path + '.jit', ModelSaveMethod.JIT, (1, 3, 224, 224))
            save_model(model, model_save_path + '.weight', ModelSaveMethod.WEIGHT)
            logger.info('save best model in {}'.format(model_save_path))
        logger.info('epoch {}, train loss {:.6f}, AUC: {:.6f}, best AUC: {:.6f}'.format(epoch_index, train_loss, roc_auc, best_roc_auc))
            
        if epoch_index < init_epoch_num:
            continue
        
        # regenerate pseudo label
        model.eval()
        Y_pred = []
        for batch_index, (data, target) in enumerate(test_dataloader):
            with torch.no_grad():
                data = data.to(device)
                output = model(data)
                Y_pred += [output]
        Y_pred = torch.cat(Y_pred, dim=0).reshape(-1).cpu().detach().numpy()
        anomaly_scores += Y_pred
        
        A_video_tensor, A_frames_index, N_video_tensor, N_frames_index = split_A_N_by_score(all_video_tensor, anomaly_scores)
        A_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[A_frames_index] == 1) / len(A_frames_index)
        N_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[N_frames_index] == 0) / len(N_frames_index)
        logger.info('cur train dataset | A correct rate: {:.6f}, N correct rate: {:.6f}'.format(A_correct_rate, N_correct_rate))
        
        X = torch.cat([A_video_tensor, N_video_tensor], dim=0).cpu()
        Y = torch.cat([
            torch.ones(A_video_tensor.size()[0]), 
            torch.zeros(N_video_tensor.size()[0])
        ]).unsqueeze(dim=1).cpu()
        
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=8)
        
        logger.info('updated A and N frames')
        
    logger.info('{} - {}'.format(model_save_path, best_roc_auc))
        
    return best_roc_auc

def train_efficientnet(model, all_video_raw_info_path, AN_init_info_path, model_save_path, epoch_num, batch_size=128, lr=0.001, img_size=224, device='cuda',
          init_epoch_num=0):
    logger.info('loading data...')
    ensure_dir(model_save_path)
    
    # data loading...
    all_video_tensor, all_video_frames_label = torch.load(all_video_raw_info_path)
    A_video_tensor, A_frames_index, N_video_tensor, N_frames_index = torch.load(AN_init_info_path)
    A_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[A_frames_index] == 1) / len(A_frames_index)
    N_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[N_frames_index] == 0) / len(N_frames_index)
    logger.info('cur train dataset | A correct rate: {:.6f}, N correct rate: {:.6f}'.format(A_correct_rate, N_correct_rate))
    
    X = torch.cat([A_video_tensor, N_video_tensor], dim=0).cpu()
    Y = torch.cat([
        torch.ones(A_video_tensor.size()[0]), 
        torch.zeros(N_video_tensor.size()[0])
    ]).unsqueeze(dim=1).cpu()
    anomaly_scores = np.zeros(all_video_tensor.size()[0])
    
    print(X.size(), Y.size(), all_video_tensor.size())
    
    # # train loader
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    dataset = MyTensorsDataset(X, Y, transforms=data_transform)
    # dataset = TensorDataset(X, Y)
    # print(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=8)
    
    # test loader
    test_dataset = TensorDataset(all_video_tensor.cpu(), all_video_frames_label.cpu())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=8)
    
    bn_dataset = TensorDataset(all_video_tensor.cpu(), all_video_frames_label.cpu())
    bn_dataloader = DataLoader(bn_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=8)
    
    # model = AnomalyScoreLearner().to(device)
    model = model.to(device)
    save_model(model, model_save_path, ModelSaveMethod.FULL, None)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # epoch_num = 50
    criterion = torch.nn.L1Loss()
    
    model.train()
    best_roc_auc = 0
    
    logger.info('training...')
    import copy
    first_trainloader = copy.deepcopy(dataloader)
    for epoch_index in range(epoch_num):
        train_loss = 0.
        
        # model.train()
        
        for batch_index, (data, target) in enumerate(dataloader):
            # print(data)
            # exit(0)
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        train_loss /= len(dataloader.dataset)
        # logger.info('epoch {}, train loss {:.6f}'.format(epoch_index, train_loss))
        
        if isnan(train_loss):
            logger.warn('loss exploded')
            return 0.5
        
        # reset_model_BN(model, first_trainloader)
        # reset_model_BN(model, bn_dataloader)
        model.eval()
        reset_running_BN(model, first_trainloader)
        
        roc_auc, auprc, f1_res = test(model, test_dataloader, all_video_frames_label, device)
        # logger.info('epoch {}, train loss {:.6f}, AUC: {:.6f}, best AUC: {:.6f}'.format(epoch_index, train_loss, roc_auc, best_roc_auc))

        # save_model(model, './raw-models/ucsd-ped1-model-{}.pth'.format(epoch_index), None, ModelSaveMethod.WEIGHT)
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            save_model(model, model_save_path, ModelSaveMethod.FULL, None)
            logger.info('save best model in {}'.format(model_save_path))
        logger.info('epoch {}, train loss {:.6f}, AUC: {:.6f}, best AUC: {:.6f}'.format(epoch_index, train_loss, roc_auc, best_roc_auc))
            
        if epoch_index < init_epoch_num:
            continue
        
        # regenerate pseudo label
        model.eval()
        Y_pred = []
        for batch_index, (data, target) in enumerate(test_dataloader):
            with torch.no_grad():
                data = data.to(device)
                output = model(data)
                Y_pred += [output]
        Y_pred = torch.cat(Y_pred, dim=0).reshape(-1).cpu().detach().numpy()
        anomaly_scores += Y_pred
        
        A_video_tensor, A_frames_index, N_video_tensor, N_frames_index = split_A_N_by_score(all_video_tensor, anomaly_scores)
        A_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[A_frames_index] == 1) / len(A_frames_index)
        N_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[N_frames_index] == 0) / len(N_frames_index)
        logger.info('cur train dataset | A correct rate: {:.6f}, N correct rate: {:.6f}'.format(A_correct_rate, N_correct_rate))
        
        X = torch.cat([A_video_tensor, N_video_tensor], dim=0).cpu()
        Y = torch.cat([
            torch.ones(A_video_tensor.size()[0]), 
            torch.zeros(N_video_tensor.size()[0])
        ]).unsqueeze(dim=1).cpu()

        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        dataset = MyTensorsDataset(X, Y, transforms=data_transform)
        # dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=8)
        
        logger.info('updated A and N frames')
        
    logger.info('{} - {}'.format(model_save_path, best_roc_auc))
        
    return best_roc_auc

public_test_loader = None
public_all_video_frames_label = None


import random
def train_us_net(model, all_video_raw_info_path, AN_init_info_path, model_save_path, epoch_num, batch_size=128, lr=0.001, device='cuda',
                 train_width_mult_list=[], train_sample_net_num=4, cal=False, init_epoch_num=0):
    logger.info('loading data...')
    ensure_dir(model_save_path)
    
    # data loading...
    all_video_tensor, all_video_frames_label = torch.load(all_video_raw_info_path)
    A_video_tensor, A_frames_index, N_video_tensor, N_frames_index = torch.load(AN_init_info_path)
    A_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[A_frames_index] == 1) / len(A_frames_index)
    N_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[N_frames_index] == 0) / len(N_frames_index)
    logger.info('cur train dataset | A correct rate: {:.6f}, N correct rate: {:.6f}'.format(A_correct_rate, N_correct_rate))
    
    X = torch.cat([A_video_tensor, N_video_tensor], dim=0).cpu()
    Y = torch.cat([
        torch.ones(A_video_tensor.size()[0]), 
        torch.zeros(N_video_tensor.size()[0])
    ]).unsqueeze(dim=1).cpu()
    anomaly_scores = np.zeros(all_video_tensor.size()[0])
    
    # train loader
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=8)
    
    # test loader
    test_dataset = TensorDataset(all_video_tensor.cpu(), all_video_frames_label.cpu())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=8)
    
    public_test_loader = test_dataloader
    public_all_video_frames_label = all_video_frames_label
    
    
    
    # model = AnomalyScoreLearner().to(device)
    model = model.to(device)
    save_model(model, model_save_path, ModelSaveMethod.FULL)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # epoch_num = 50
    criterion = torch.nn.L1Loss()
    
    model.train()
    best_roc_auc = 0
    
    logger.info('training...')
    
    for epoch_index in range(epoch_num):
        train_loss = 0.
        
        # model.train()
        # for name, module in model.named_modules():
        #     if isinstance(module, USBatchNorm2d):
        #         module.track_running_stats = True
        
        for batch_index, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # if cal:
            #     model(data)
            #     continue
            
            min_width, max_width = min(train_width_mult_list), max(train_width_mult_list)
            widths_train = []
            for _ in range(train_sample_net_num - 2):
                widths_train.append(
                    random.uniform(min_width, max_width))
                
            widths_train = [max_width, min_width] + widths_train
            # logger.info('sampled width: {}'.format(widths_train))

            
            optimizer.zero_grad()
            for width_mult in widths_train:
                # print(width_mult)
                # width_mult = round(width_mult, 3)
                # print(width_mult)
                model.apply(lambda m: setattr(m, 'width_mult', width_mult))
            
                output = model(data)
                loss = criterion(output, target)
                train_loss += loss.item()
                loss.backward()
            optimizer.step()
            
        model.feature_learner.apply(lambda m: setattr(m, 'width_mult', 1.0))
            
        train_loss /= len(dataloader.dataset)
        # logger.info('epoch {}, train loss {:.6f}'.format(epoch_index, train_loss))
        
        if isnan(train_loss):
            logger.warn('loss exploded')
            return 0.5
        
        roc_auc, auprc, f1_res = test(model, test_dataloader, all_video_frames_label, device)
        # logger.info('epoch {}, train loss {:.6f}, AUC: {:.6f}, best AUC: {:.6f}'.format(epoch_index, train_loss, roc_auc, best_roc_auc))

        # save_model(model, './raw-models/ucsd-ped1-model-{}.pth'.format(epoch_index), None, ModelSaveMethod.WEIGHT)
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            save_model(model, model_save_path, ModelSaveMethod.FULL)
            logger.info('save best model in {}'.format(model_save_path))
        logger.info('epoch {}, train loss {:.6f}, AUC: {:.6f}, best AUC: {:.6f}'.format(epoch_index, train_loss, roc_auc, best_roc_auc))
            
        if epoch_index < init_epoch_num:
            continue
        
        # regenerate pseudo label
        model.eval()
        Y_pred = []
        for batch_index, (data, target) in enumerate(test_dataloader):
            with torch.no_grad():
                data = data.to(device)
                output = model(data)
                Y_pred += [output]
        Y_pred = torch.cat(Y_pred, dim=0).reshape(-1).cpu().detach().numpy()
        anomaly_scores += Y_pred
        
        A_video_tensor, A_frames_index, N_video_tensor, N_frames_index = split_A_N_by_score(all_video_tensor, anomaly_scores)
        A_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[A_frames_index] == 1) / len(A_frames_index)
        N_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[N_frames_index] == 0) / len(N_frames_index)
        logger.info('cur train dataset | A correct rate: {:.6f}, N correct rate: {:.6f}'.format(A_correct_rate, N_correct_rate))
        
        X = torch.cat([A_video_tensor, N_video_tensor], dim=0).cpu()
        Y = torch.cat([
            torch.ones(A_video_tensor.size()[0]), 
            torch.zeros(N_video_tensor.size()[0])
        ]).unsqueeze(dim=1).cpu()
        
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=8)
        
        logger.info('updated A and N frames')
        
    logger.info('{} - {}'.format(model_save_path, best_roc_auc))
    
    if cal:
        from zedl.common.data_record import CSVDataRecord
        sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/slimmable_networks')
        from open_api.us_net import convert_model_to_us_net, set_us_net_width_mult, export_jit_us_sub_net, bn_calibration_init, set_FLAGS
        import pandas as pd

        model = torch.load(model_save_path)
        
        # print(model)
        
        logger.info('before bn cal')
        for w in [train_width_mult_list[0], train_width_mult_list[-1]]:
            model.feature_learner.apply(lambda m: setattr(m, 'width_mult', w))
            auc, _, _ = test(model, test_dataloader, all_video_frames_label, 'cuda')
            logger.info('width mult: {}, AUC: {:.6f}'.format(w, auc))
            
        model.feature_learner.apply(lambda m: setattr(m, 'width_mult', 1.0))
        model.feature_learner.apply(bn_calibration_init)
        
        # all_video_tensor, all_video_frames_label = torch.load(all_video_raw_info_path)
        # A_video_tensor, A_frames_index, N_video_tensor, N_frames_index = torch.load(AN_init_info_path)
        # A_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[A_frames_index] == 1) / len(A_frames_index)
        # N_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[N_frames_index] == 0) / len(N_frames_index)
        # logger.info('cur train dataset | A correct rate: {:.6f}, N correct rate: {:.6f}'.format(A_correct_rate, N_correct_rate))
        
        # X = torch.cat([A_video_tensor, N_video_tensor], dim=0).cpu()
        # Y = torch.cat([
        #     torch.ones(A_video_tensor.size()[0]), 
        #     torch.zeros(N_video_tensor.size()[0])
        # ]).unsqueeze(dim=1).cpu()
        # anomaly_scores = np.zeros(all_video_tensor.size()[0])
        
        # # train loader
        # dataset = TensorDataset(X, Y)
        # raw_dataloader = DataLoader(dataset, batch_size=batch_size, 
        #                         shuffle=True, num_workers=8)
        
        all_dataset = TensorDataset(all_video_tensor)
        all_dataloader = DataLoader(all_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=8)
        
        model.feature_learner.train()
        for batch_index, (data, ) in enumerate(all_dataloader):
            data = data.to(device)
            for w in train_width_mult_list[::-1]:
                model.feature_learner.apply(lambda m: setattr(m, 'width_mult', w))
                model(data)
            
        logger.info('after bn cal')
        acc_record = CSVDataRecord(model_save_path + '.acc-record', ['width_mult', 'acc'], backup=False)
        for w in train_width_mult_list:
            model.feature_learner.apply(lambda m: setattr(m, 'width_mult', w))
            auc, _, _ = test(model, test_dataloader, all_video_frames_label, 'cuda')
            logger.info('width mult: {}, AUC: {:.6f}'.format(w, auc))
            
            acc_record.write([w, auc])
            
        acc_data = pd.read_csv(model_save_path + '.acc-record')
        plt.plot(acc_data['width_mult'], acc_data['acc'])
        plt.ylabel('AUC')
        plt.xlabel('width mult (active channel ratio)')
        plt.savefig(model_save_path + '.acc-record.png', dpi=300)
        plt.clf()
    
    # if return_dataloader:
        
    #     return public_test_loader, public_all_video_frames_label
        
    return best_roc_auc


sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/fn3')
from fn3_channel_open_api.fn3_channel import set_fn3_channel_channels, export_active_sub_net
def train_fn3_channel(model, all_video_raw_info_path, AN_init_info_path, model_save_path, epoch_num, batch_size=128, lr=0.001, device='cuda',
                      fn3_channel_fl_key_layers_info=[], disabled_layers=[], active_channel_ratio_list=[], init_epoch_num=0):
    logger.info('loading data...')
    ensure_dir(model_save_path)
    
    # data loading...
    all_video_tensor, all_video_frames_label = torch.load(all_video_raw_info_path)
    A_video_tensor, A_frames_index, N_video_tensor, N_frames_index = torch.load(AN_init_info_path)
    A_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[A_frames_index] == 1) / len(A_frames_index)
    N_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[N_frames_index] == 0) / len(N_frames_index)
    logger.info('cur train dataset | A correct rate: {:.6f}, N correct rate: {:.6f}'.format(A_correct_rate, N_correct_rate))
    
    X = torch.cat([A_video_tensor, N_video_tensor], dim=0).cpu()
    Y = torch.cat([
        torch.ones(A_video_tensor.size()[0]), 
        torch.zeros(N_video_tensor.size()[0])
    ]).unsqueeze(dim=1).cpu()
    anomaly_scores = np.zeros(all_video_tensor.size()[0])
    
    # train loader
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=8)
    
    # test loader
    test_dataset = TensorDataset(all_video_tensor.cpu(), all_video_frames_label.cpu())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=8)
    
    public_test_loader = test_dataloader
    public_all_video_frames_label = all_video_frames_label
    
    # model = AnomalyScoreLearner().to(device)
    model = model.to(device)
    save_model(model, model_save_path, ModelSaveMethod.FULL)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # epoch_num = 50
    criterion = torch.nn.L1Loss()
    
    model.train()
    best_roc_auc = 0
    
    logger.info('training...')
    
    for epoch_index in range(epoch_num):
        train_loss = 0.
        
        # model.train()
        
        for batch_index, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            fn3_channel_layers_name = [i[0] for i in fn3_channel_fl_key_layers_info]
            fn3_channel_channels = [i[1] for i in fn3_channel_fl_key_layers_info]
            
            for i, c in enumerate(fn3_channel_channels):
                if fn3_channel_layers_name[i] in disabled_layers:
                    continue
                fn3_channel_channels[i] = random.randint(1, c)
            
            set_fn3_channel_channels(model.feature_learner, fn3_channel_channels)
            
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        set_fn3_channel_channels(model.feature_learner, [i[1] for i in fn3_channel_fl_key_layers_info])
            
        train_loss /= len(dataloader.dataset)
        # logger.info('epoch {}, train loss {:.6f}'.format(epoch_index, train_loss))
        
        if isnan(train_loss):
            logger.warn('loss exploded')
            return 0.5
        
        roc_auc, auprc, f1_res = test(model, test_dataloader, all_video_frames_label, device)
        # logger.info('epoch {}, train loss {:.6f}, AUC: {:.6f}, best AUC: {:.6f}'.format(epoch_index, train_loss, roc_auc, best_roc_auc))

        # save_model(model, './raw-models/ucsd-ped1-model-{}.pth'.format(epoch_index), None, ModelSaveMethod.WEIGHT)
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            save_model(model, model_save_path, ModelSaveMethod.FULL)
            logger.info('save best model in {}'.format(model_save_path))
        logger.info('epoch {}, train loss {:.6f}, AUC: {:.6f}, best AUC: {:.6f}'.format(epoch_index, train_loss, roc_auc, best_roc_auc))
            
        if epoch_index < init_epoch_num:
            continue
        
        # regenerate pseudo label
        model.eval()
        Y_pred = []
        for batch_index, (data, target) in enumerate(test_dataloader):
            with torch.no_grad():
                data = data.to(device)
                output = model(data)
                Y_pred += [output]
        Y_pred = torch.cat(Y_pred, dim=0).reshape(-1).cpu().detach().numpy()
        anomaly_scores += Y_pred
        
        A_video_tensor, A_frames_index, N_video_tensor, N_frames_index = split_A_N_by_score(all_video_tensor, anomaly_scores)
        A_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[A_frames_index] == 1) / len(A_frames_index)
        N_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[N_frames_index] == 0) / len(N_frames_index)
        logger.info('cur train dataset | A correct rate: {:.6f}, N correct rate: {:.6f}'.format(A_correct_rate, N_correct_rate))
        
        X = torch.cat([A_video_tensor, N_video_tensor], dim=0).cpu()
        Y = torch.cat([
            torch.ones(A_video_tensor.size()[0]), 
            torch.zeros(N_video_tensor.size()[0])
        ]).unsqueeze(dim=1).cpu()
        
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=8)
        
        logger.info('updated A and N frames')
        
    logger.info('{} - {}'.format(model_save_path, best_roc_auc))
    
    from zedl.common.data_record import CSVDataRecord
    import pandas as pd

    model = torch.load(model_save_path)
    
    acc_record = CSVDataRecord(model_save_path + '.acc-record', ['active_channel_ratio', 'acc'], backup=False)
    for r in active_channel_ratio_list:
        active_channels = [int(i[1] * (r if i[0] not in disabled_layers else 1))
                           for i in fn3_channel_fl_key_layers_info]
        set_fn3_channel_channels(model.feature_learner, active_channels)
        
        auc, _, _ = test(model, test_dataloader, all_video_frames_label, 'cuda')
        logger.info('active_channel_ratio: {}, AUC: {:.6f}'.format(r, auc))
        
        acc_record.write([r, auc])
        
    acc_data = pd.read_csv(model_save_path + '.acc-record')
    plt.plot(acc_data['active_channel_ratio'], acc_data['acc'])
    plt.ylabel('AUC')
    plt.xlabel('active channel ratio')
    plt.savefig(model_save_path + '.acc-record.png', dpi=300)
    plt.clf()
    
        
    return best_roc_auc


def test_cgnet(model, test_dataloader, all_video_frames_label, device='cuda'):
    model.eval()
    Y_pred = []
    
    batch_num = 0
    cgnet_flops_reduce_ratio = 0
    for batch_index, (data, target) in enumerate(test_dataloader):
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            
            output = model.module(data)
            Y_pred += [output]
            
            batch_num += 1
            # cgnet_flops_reduce_ratio += get_cgnet_flops_save_ratio(model, data)
            raw_flops = 0
            final_flops = 0
            
            for name, module in model.module.named_modules():
                if isinstance(module, CGConv2d):
                    raw_flops += module.num_out
                    final_flops += module.num_full
                    
            cgnet_flops_reduce_ratio += (raw_flops - final_flops) / raw_flops
        
    cgnet_flops_reduce_ratio /= batch_num
    
    Y_pred = torch.cat(Y_pred, dim=0)
    
    labels = all_video_frames_label.cpu()
    scores = Y_pred.cpu()
    
    threshold = 0.50
    scores[scores >= threshold] = 1
    scores[scores <  threshold] = 0
    
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    
    f1_res = f1_score(labels, scores)
    
    return roc_auc, average_precision_score(labels, scores), f1_res, cgnet_flops_reduce_ratio


sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/cgnet')
from cgnet_open_api import add_cgnet_loss, get_cgnet_flops_save_ratio
def train_cgnet(model, all_video_raw_info_path, AN_init_info_path, model_save_path, epoch_num, batch_size=128, lr=0.001, device='cuda',
                gtar=0, init_epoch_num=0):
    logger.info('loading data...')
    ensure_dir(model_save_path)
    
    # data loading...
    all_video_tensor, all_video_frames_label = torch.load(all_video_raw_info_path)
    A_video_tensor, A_frames_index, N_video_tensor, N_frames_index = torch.load(AN_init_info_path)
    A_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[A_frames_index] == 1) / len(A_frames_index)
    N_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[N_frames_index] == 0) / len(N_frames_index)
    logger.info('cur train dataset | A correct rate: {:.6f}, N correct rate: {:.6f}'.format(A_correct_rate, N_correct_rate))
    
    X = torch.cat([A_video_tensor, N_video_tensor], dim=0).cpu()
    Y = torch.cat([
        torch.ones(A_video_tensor.size()[0]), 
        torch.zeros(N_video_tensor.size()[0])
    ]).unsqueeze(dim=1).cpu()
    anomaly_scores = np.zeros(all_video_tensor.size()[0])
    
    # train loader
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=8)
    
    # test loader
    test_dataset = TensorDataset(all_video_tensor.cpu(), all_video_frames_label.cpu())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=8)
    
    public_test_loader = test_dataloader
    public_all_video_frames_label = all_video_frames_label
    
    # model = AnomalyScoreLearner().to(device)
    model = model.to(device)
    save_model(model, model_save_path, ModelSaveMethod.FULL)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # epoch_num = 50
    criterion = torch.nn.L1Loss()
    
    model.train()
    best_roc_auc = 0
    
    train_record = CSVDataRecord(model_save_path + '.train-record', ['epoch', 'auc', 'sparsity'])
    
    logger.info('training...')
    
    for epoch_index in range(epoch_num):
        train_loss = 0.
        
        model.train()
        
        for batch_index, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            add_cgnet_loss(model, loss, gtar)
            
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        train_loss /= len(dataloader.dataset)
        # logger.info('epoch {}, train loss {:.6f}'.format(epoch_index, train_loss))
        
        if isnan(train_loss):
            logger.warn('loss exploded')
            return 0.5
        
        roc_auc, auprc, f1_res, cgnet_flops_save_ratio = test_cgnet(model, test_dataloader, all_video_frames_label, device)
        train_record.write([epoch_index, roc_auc, cgnet_flops_save_ratio])
        # logger.info('epoch {}, train loss {:.6f}, AUC: {:.6f}, best AUC: {:.6f}'.format(epoch_index, train_loss, roc_auc, best_roc_auc))

        # save_model(model, './raw-models/ucsd-ped1-model-{}.pth'.format(epoch_index), None, ModelSaveMethod.WEIGHT)
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            save_model(model, model_save_path, ModelSaveMethod.FULL)
            write_json(model_save_path + '.metrics', {
                'auc': best_roc_auc,
                'sparsity': cgnet_flops_save_ratio
            }, backup=False)
            logger.info('save best model in {}'.format(model_save_path))
        logger.info('epoch {}, train loss {:.6f}, AUC: {:.6f}, FLOPs save {:.2f}%, '
                    'best AUC: {:.6f}'.format(epoch_index, train_loss, roc_auc, cgnet_flops_save_ratio * 100., best_roc_auc))
            
        if epoch_index < init_epoch_num:
            continue
        
        # regenerate pseudo label
        model.eval()
        Y_pred = []
        for batch_index, (data, target) in enumerate(test_dataloader):
            with torch.no_grad():
                data = data.to(device)
                output = model(data)
                Y_pred += [output]
        Y_pred = torch.cat(Y_pred, dim=0).reshape(-1).cpu().detach().numpy()
        anomaly_scores += Y_pred
        
        A_video_tensor, A_frames_index, N_video_tensor, N_frames_index = split_A_N_by_score(all_video_tensor, anomaly_scores)
        A_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[A_frames_index] == 1) / len(A_frames_index)
        N_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[N_frames_index] == 0) / len(N_frames_index)
        logger.info('cur train dataset | A correct rate: {:.6f}, N correct rate: {:.6f}'.format(A_correct_rate, N_correct_rate))
        
        X = torch.cat([A_video_tensor, N_video_tensor], dim=0).cpu()
        Y = torch.cat([
            torch.ones(A_video_tensor.size()[0]), 
            torch.zeros(N_video_tensor.size()[0])
        ]).unsqueeze(dim=1).cpu()
        
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=8)
        
        logger.info('updated A and N frames')
        
    logger.info('{} - {}'.format(model_save_path, best_roc_auc))
        
    return best_roc_auc

