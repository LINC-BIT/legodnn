from math import isnan
import sys

sys.path.insert(0, '/data/zql/zedl')
from zedl.common.data_record import CSVDataRecord, write_json
sys.path.insert(0, '../../')

import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from functools import reduce
from sklearn.metrics import auc, roc_curve, f1_score, precision_score, recall_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler

# from model.ornet.util.datasets import UCSDDataset, get_extracted_features
from cv_task.anomaly_detection.methods.ornet.init_anomaly_detection import split_A_N, split_A_N_by_score
from cv_task.anomaly_detection.methods.ornet.anomaly_score_learner import AnomalyScoreLearner

sys.path.insert(0, '/data/zql/zedl')
from zedl.common.log import logger
from zedl.dl.common.model import save_model, ModelSaveMethod
from zedl.dl.common.env import set_random_seed
from zedl.common.file import ensure_dir

import numpy as np

# sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/once-for-all')
# from ofa_open_api.ofa_ornet import reset_running_BN, sample_ofa_sub_net_width_mult, set_ofa_sub_net_width_mult

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
    
    return roc_auc, average_precision_score(labels, scores), f1_res


def list_sum(x):
    	return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])

def list_mean(x):
	return list_sum(x) / len(x)

def test_mult_ofa_sub_net(model, init_dataloader, test_dataloader, all_video_frames_label, depth_list, expand_ratio_list, width_mult_list, device='cuda'):
    model.eval()
    auc_of_subnet = []
    
    depth_list.sort()
    width_mult_list.sort()
    expand_ratio_list.sort()
    
    for d in depth_list:
        for w_idx in range(len(width_mult_list)):
            for e in expand_ratio_list:
                model.feature_learner.set_active_subnet(d=d, e=e, w=w_idx)
                logger.info("reset BN Statistics")
                reset_running_BN(model, init_dataloader)
                roc_auc, auprc, f1_res = test(model, test_dataloader, all_video_frames_label)
                logger.info('D{}-W{}-E{} AUC: {:.6f}'.format(d, width_mult_list[w_idx], e, roc_auc))
                auc_of_subnet.append(roc_auc)
    return list_mean(auc_of_subnet)


def train_ofa(model, all_video_raw_info_path, AN_init_info_path, model_save_path, epoch_num, depth_list, expand_ratio_list, width_mult_list,train_phase, init_dataloader, batch_size=128, lr=0.001, device='cuda', init_epoch_num=0, fl_name='resnet18'):
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
    
    dataloader = init_dataloader
    
    if train_phase!='raw' and train_phase!='phase_{}'.format(1.0):
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


    # test_mult_ofa_sub_net(model, init_dataloader, test_dataloader, all_video_frames_label, depth_list, expand_ratio_list, width_mult_list, device)
    # import copy
    # first_dataloader = 
    # model = AnomalyScoreLearner().to(device)
    model = model.to(device)
    save_model(model, model_save_path, None, ModelSaveMethod.FULL)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # epoch_num = 50
    criterion = torch.nn.L1Loss()
    
    model.train()
    best_roc_auc = 0
    
    logger.info('training...')
    
    for epoch_index in range(epoch_num):
        train_loss = 0.
        
        # model.train()
        
        from tqdm import tqdm
        for batch_index, (data, target) in enumerate(tqdm(dataloader)):
            data, target = data.to(device), target.to(device)

            # TODO: OFA
            if fl_name=='resnet18' or fl_name=='resnet50':
                model.feature_learner.sample_active_subnet()
            elif fl_name=='alexnet' or fl_name=='vgg16':
                sample_ofa_sub_net_width_mult(model.feature_learner, width_mult_list)
            else:
                raise NotImplementedError
            
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
        
        
        if fl_name=='resnet18' or fl_name=='resnet50':
            model.feature_learner.set_active_subnet(d=max(depth_list), e=max(expand_ratio_list), w=len(width_mult_list)-1)
        elif fl_name=='alexnet' or fl_name=='vgg16':
            set_ofa_sub_net_width_mult(model.feature_learner, max(width_mult_list))
        else:
            raise NotImplementedError
        roc_auc, auprc, f1_res = test(model, test_dataloader, all_video_frames_label, device)
        # roc_auc = test_mult_ofa_sub_net(model, init_dataloader, test_dataloader, all_video_frames_label, depth_list, expand_ratio_list, width_mult_list, device)
        if fl_name=='resnet18' or fl_name=='resnet50':
            model.feature_learner.set_active_subnet(d=max(depth_list), e=max(expand_ratio_list), w=len(width_mult_list)-1)
        elif fl_name=='alexnet' or fl_name=='vgg16':
            set_ofa_sub_net_width_mult(model.feature_learner, max(width_mult_list))
        else:
            raise NotImplementedError
        # logger.info('epoch {}, train loss {:.6f}, AUC: {:.6f}, best AUC: {:.6f}'.format(epoch_index, train_loss, roc_auc, best_roc_auc))

        # save_model(model, './raw-models/ucsd-ped1-model-{}.pth'.format(epoch_index), None, ModelSaveMethod.WEIGHT)
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            save_model(model, model_save_path, None, ModelSaveMethod.FULL)
            logger.info('save best model in {}'.format(model_save_path))
        logger.info('epoch {}, train loss {:.6f}, AUC: {:.6f}, best AUC: {:.6f}'.format(epoch_index, train_loss, roc_auc, best_roc_auc))
            
        if epoch_index < init_epoch_num:
            continue
        
        # regenerate pseudo label
        model.eval()
        Y_pred = []
        from tqdm import tqdm
        for batch_index, (data, target) in enumerate(tqdm(test_dataloader)):
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