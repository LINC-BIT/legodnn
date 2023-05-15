import os
import sys

import torch
sys.path.insert(0, '../../../')

from mmseg.apis import init_segmentor
from cv_task.semantic_segmentation.mmseg_models.load_mode import LOAD_MODE
from legodnn.utils.dl.common.model import get_model_size, get_module
# from cv_task.semantic_segmentation.mmseg_models.legodnn_configs import get_fcn_unet_s5_d16_64x64_40k_drive_config

pretrained_root_path = '/data/gxy/pretrained_models/mmsegmentation/'

def fcn_unet_s5_16(config, checkpoint=None, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    checkpoint = os.path.join(pretrained_root_path, 'fcn_unet_s5-d16_64x64_40k_drive_20201223_191051-5daf6d3b.pth')
    if mode=='lego_jit':
        segmentor = init_segmentor(config, checkpoint, device=device)
        segmentor.forward = segmentor.forward_dummy
        
    elif mode=='mmseg_test':
        segmentor = init_segmentor(config, checkpoint, device=device)
        segmentor = segmentor
    else:
        raise NotImplementedError
    segmentor.eval()
    return segmentor


if __name__=='__main__':
    model_config = get_fcn_unet_s5_d16_64x64_40k_drive_config(((1,3,224,224)))
    
    model = fcn_unet_s5_16(model_config)
    print(model)
    print('model size {:.3f}MB'.format(get_model_size(model) / 1024**2))
    # input = torch.rand((1,3,224,224)).cuda()
    # model(input)