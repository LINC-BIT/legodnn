
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def ornet_ucsd_ped1_dataloader(all_video_raw_info_path='/data/zql/mcw/model/ornet/ucsd-ped1-raw-data.pth', AN_init_info_path='/data/zql/mcw/model/ornet/ucsd-ped1-AN-tensor.pth', train_batch_size=64, test_batch_size=64, num_workers=8):
    # data loading...
    all_video_tensor, all_video_frames_label = torch.load(all_video_raw_info_path)
    A_video_tensor, A_frames_index, N_video_tensor, N_frames_index = torch.load(AN_init_info_path)
    A_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[A_frames_index] == 1) / len(A_frames_index)
    N_correct_rate = 1. * torch.sum(all_video_frames_label.squeeze(dim=1)[N_frames_index] == 0) / len(N_frames_index)
    print('cur train dataset | A correct rate: {:.6f}, N correct rate: {:.6f}'.format(A_correct_rate, N_correct_rate))
    
    X = torch.cat([A_video_tensor, N_video_tensor], dim=0).cpu()
    Y = torch.cat([
        torch.ones(A_video_tensor.size()[0]), 
        torch.zeros(N_video_tensor.size()[0])
    ]).unsqueeze(dim=1).cpu()
    # anomaly_scores = np.zeros(all_video_tensor.size()[0])
    
    # train loader
    train_dataset = TensorDataset(X, Y)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, 
                            shuffle=True, num_workers=8)
    
    # test loader
    test_dataset = TensorDataset(all_video_tensor.cpu(), all_video_frames_label.cpu())
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, 
                                shuffle=False, num_workers=8)
    
    # all_dataset = TensorDataset(all_video_tensor.cpu(), all_video_frames_label.cpu())
    # all_dataloader = DataLoader(all_dataset, batch_size=batch_size, 
    #                             shuffle=True, num_workers=8)
    return train_dataloader, test_dataloader, all_video_frames_label