import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tqdm
from typing import List

from cv_task.anomaly_detection.methods.ornet.init_anomaly_detection.iforest import iforest
from cv_task.anomaly_detection.methods.ornet.init_anomaly_detection.sp import sp

import sys
sys.path.insert(0, '/data/zql/zedl')
from zedl.common.log import logger


def split_A_N_by_score(video_tensor: torch.Tensor, frames_anomaly_score, device='cuda'):
    frame_num = video_tensor.size()[0]
    
    A_frames_index = np.argsort(-frames_anomaly_score)[0: frame_num // 10]
    N_frames_index = np.argsort(frames_anomaly_score)[0: frame_num // 5]
    
    A_video_tensor = video_tensor[A_frames_index].to(device)
    N_video_tensor = video_tensor[N_frames_index].to(device)

    logger.info('A video tensor: {}, N video tensor: {}'.format(A_video_tensor.size(), N_video_tensor.size()))
    return A_video_tensor, A_frames_index, N_video_tensor, N_frames_index


def split_A_N(video_tensor: torch.Tensor, video_features_tensor: torch.Tensor, device='cuda'):
    frames_tensor = video_features_tensor.split(1, 0)
    frames_tensor = [t.squeeze() for t in frames_tensor]
    
    frames_sp_score, frames_iforest_score = [], []
    cache_iforest = None
    
    logger.info('computing anomaly score for each frame...')
    
    for frame_tensor in tqdm.tqdm(frames_tensor, total=len(frames_tensor), dynamic_ncols=True):
        sp_score = sp(frames_tensor, frame_tensor)
        iforest_score, iforest_ins = iforest(frames_tensor, frame_tensor, cache_iforest=cache_iforest)
        cache_iforest = iforest_ins
        
        frames_sp_score += [sp_score]
        frames_iforest_score += [iforest_score]
    
    frames_sp_score = np.asarray(frames_sp_score).reshape((-1, 1))
    frames_iforest_score = np.asarray(frames_iforest_score).reshape((-1, 1))
    
    frames_sp_score = MinMaxScaler().fit_transform(frames_sp_score).reshape(-1)
    frames_iforest_score = MinMaxScaler().fit_transform(frames_iforest_score).reshape(-1)
    frames_iforest_score = 1. - frames_iforest_score
    
    avg_score = (frames_sp_score + frames_iforest_score) / 2
    
    return split_A_N_by_score(video_tensor, avg_score, device)
