import os
import sys
sys.path.insert(0, '../../../')

from mmseg.apis import init_segmentor
from cv_task.semantic_segmentation.mmseg_models.load_mode import LOAD_MODE
from legodnn.utils.dl.common.model import get_model_size, get_module

pretrained_root_path = '/data/gxy/pretrained_models/mmsegmentation/'

def ccnet_r50_d8(config, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    checkpoint = os.path.join(pretrained_root_path, 'ccnet_r50-d8_512x1024_80k_cityscapes_20200617_010421-869a3423.pth')
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
    model = ccnet_r50_d8('/data/gxy/legodnn-public-version_semantic_segmentation/cv_task/semantic_segmentation/mmseg_models/legodnn_configs/ccnet_r50-d8_512x1024_80k_cityscapes.py')
    print(model)
    print('model size {:.3f}MB'.format(get_model_size(model) / 1024**2))
    print('backbone size {:.3f}MB'.format(get_model_size(get_module(model, 'backbone')) / 1024**2))
    # print('neck size {:.3f}MB'.format(get_model_size(get_module(model, 'neck')) / 1024**2))
    print('decode_head size {:.3f}MB'.format(get_model_size(get_module(model, 'decode_head')) / 1024**2))
    print('auxiliary_head size {:.3f}MB'.format(get_model_size(get_module(model, 'auxiliary_head')) / 1024**2))


