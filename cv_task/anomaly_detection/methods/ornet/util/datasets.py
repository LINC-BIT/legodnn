from util.video_loader import *
import glob
import os
import torchvision
from torchvision.models import resnet50
import torch
import torch.nn
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import numpy as np
from sklearn.decomposition import PCA
import re

from util.log import logger
from util.model import get_module


def get_UCSD_ped_all_videos_path(ped_root_dir, max_train_video_num=None, max_test_video_num=None):
    res = []
    
    test_dir_path = os.path.join(ped_root_dir, 'Test')
    train_dir_path = os.path.join(ped_root_dir, 'Train')
    
    train_video_frames_dir_path = glob.glob(os.path.join(train_dir_path, 'Train???'))
    test_video_frames_dir_path = glob.glob(os.path.join(test_dir_path, 'Test???'))
    
    res += train_video_frames_dir_path if max_train_video_num is None else train_video_frames_dir_path[:max_train_video_num]
    res += test_video_frames_dir_path if max_test_video_num is None else test_video_frames_dir_path[:max_test_video_num]
    
    res.sort()
    return res


def get_USCD_ped_anomaly_frames_file_path(root_dir='/data/zql/datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1'):
    res = []
    
    gt_info_file_path = glob.glob(os.path.join(root_dir, './Test/*.m'))[0]
    with open(gt_info_file_path, 'r') as f:
        lines = f.readlines()
        regx = re.compile('(\\d+):(\\d+)')
        
        for test_video_index, line in enumerate(lines[1:]):
            info_list = regx.findall(line)
            for info in info_list:
                start_frame_index, end_frame_index = info
                start_frame_index, end_frame_index = int(start_frame_index), int(end_frame_index)    

                res += [os.path.join(root_dir, 'Test/Test{:0>3d}/{:0>3d}.tif'.format(test_video_index, i)) 
                        for i in range(start_frame_index, end_frame_index)]
            
    return res
    

# sample size: (ic, time, isize, isize)
def UCSDDataset(root_dir='/data/zql/datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1', img_size=224, 
                max_train_video_num=None, max_test_video_num=None):
    
    return VideoDataset(get_UCSD_ped_all_videos_path(root_dir, max_train_video_num, max_test_video_num), 
                                transform=torchvision.transforms.Compose([
                                    VideoFolderPathToTensor(get_USCD_ped_anomaly_frames_file_path(root_dir)),
                                    VideoResize([img_size, img_size]),
                                ]))


def get_extracted_features(all_video_tensor: torch.Tensor, batch_size=128, device='cuda'):
    from model.ornet.pretrained_model.model import ft_net
    
    class FeatureExtractorNN(torch.nn.Module):
        def __init__(self):
            super(FeatureExtractorNN, self).__init__()
            
            self.base_model = ft_net(751)
            self.base_model.load_state_dict(torch.load('./pretrained_model/net_last.pth'))
            
            self.layers = self._get_extractor_layers_of_base_model()
            
        def _get_extractor_layers_of_base_model(self):
            return [
                self.base_model
            ]
            
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
                
            return x
    
    logger.info('all_video_tensor shape: {}'.format(all_video_tensor.size()))
    
    logger.info('extract features via NN...')

    feature_extractor_nn = FeatureExtractorNN().to(device)
    tmp_dataset = TensorDataset(all_video_tensor)
    tmp_dataloader = DataLoader(tmp_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=4)
    
    extracted_tensors = []
    for batch in tqdm.tqdm(tmp_dataloader, total=all_video_tensor.size()[0] // batch_size, dynamic_ncols=True):
        batch_frames, = batch
        batch_frames = batch_frames.to(device)
        extracted_tensor = feature_extractor_nn(batch_frames)
        extracted_tensors += [extracted_tensor.cpu().detach().numpy()]
        
    extracted_features = np.vstack(extracted_tensors)
    logger.info('extracted features shape: {}'.format(extracted_features.shape))
    
    logger.info('PCA...')
    pca = PCA(n_components=100)
    pcaed_features = pca.fit_transform(extracted_features)
    logger.info('PCAed features shape: {}, PCA variance top-10 ratio: {}'.format(pcaed_features.shape, 
                                                                                 pca.explained_variance_ratio_[0:10]))
    pcaed_features = torch.from_numpy(pcaed_features)

    return pcaed_features
