import torch
from torchvision.models.resnet import resnet18
from cv_task.anomaly_detection.methods.ornet.anomaly_score_learner import AnomalyScoreLearner
import torch
import numpy as np

def ornet_resnet18_ucsd_ped1(pretrained=True, device='cuda'):
    if pretrained:
        model = torch.load('/data/gxy/legodnn-auto-on-cv-models/cv_task_model/anomaly_detection/ornet_resnet18_ucsd_ped1/ornet_resnet18_ucsd_ped1').to(device)
    else:
        backbone = resnet18(pretrained=True).to(device)
        model = AnomalyScoreLearner(backbone, 1000).to(device)
    return model

def get_ornet_ucsd_ped1_dataloader(all_video_raw_info_path='/data/zql/mcw/model/ornet/ucsd-ped1-raw-data.pth', AN_init_info_path='/data/zql/mcw/model/ornet/ucsd-ped1-AN-tensor.pth', batch_size=128):
    from torch.utils.data import TensorDataset, DataLoader
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
    anomaly_scores = np.zeros(all_video_tensor.size()[0])
    
    # train loader
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    
    # test loader
    test_dataset = TensorDataset(all_video_tensor.cpu(), all_video_frames_label.cpu())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=4)
    
    all_dataset = TensorDataset(all_video_tensor.cpu(), all_video_frames_label.cpu())
    all_dataloader = DataLoader(all_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4)
    return dataloader, all_dataloader, test_dataloader, all_video_frames_label


def get_bn_dataloader(model, all_video_raw_info_path='/data/zql/mcw/model/ornet/ucsd-ped1-raw-data.pth', AN_init_info_path='/data/zql/mcw/model/ornet/ucsd-ped1-AN-tensor.pth', batch_size=128, device='cuda'):
    from cv_task.anomaly_detection.methods.ornet.init_anomaly_detection import split_A_N, split_A_N_by_score
    from torch.utils.data import TensorDataset, DataLoader
    
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
    anomaly_scores = np.zeros(all_video_tensor.size()[0])
    
    # train loader
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    
    # test loader
    test_dataset = TensorDataset(all_video_tensor.cpu(), all_video_frames_label.cpu())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=4)
    
    all_dataset = TensorDataset(all_video_tensor.cpu(), all_video_frames_label.cpu())
    all_dataloader = DataLoader(all_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4)
    
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
    print('cur train dataset | A correct rate: {:.6f}, N correct rate: {:.6f}'.format(A_correct_rate, N_correct_rate))
    
    X = torch.cat([A_video_tensor, N_video_tensor], dim=0).cpu()
    Y = torch.cat([
        torch.ones(A_video_tensor.size()[0]), 
        torch.zeros(N_video_tensor.size()[0])
    ]).unsqueeze(dim=1).cpu()
    
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)

    return dataloader, all_dataloader, test_dataloader, all_video_frames_label