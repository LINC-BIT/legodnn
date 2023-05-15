import os
# import sys

# import torch
# sys.path.insert(0, '../../../')

from mmseg.apis import init_segmentor
from cv_task.semantic_segmentation.mmseg_models.load_mode import LOAD_MODE
from legodnn.utils.dl.common.model import get_model_size, get_module
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
# from cv_task.semantic_segmentation.mmseg_models.legodnn_configs import get_fcn_r18_d8_320x320_10k_cityscapes_config

pretrained_root_path = '/data/gxy/pretrained_models/mmsegmentation/'

def fcn_r18_d8_320_320(config, checkpoint=None, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    if mode=='lego_jit':
        segmentor = init_segmentor(config, checkpoint, device=device)
        segmentor.forward = segmentor.forward_dummy
    elif mode=='mmseg_test':
        segmentor = init_segmentor(config, checkpoint, device=device)
        segmentor = segmentor
    elif mode=='mmseg_train':
        segmentor = build_segmentor(config.model, train_cfg=config.get('train_cfg'), test_cfg=config.get('test_cfg'))
        if checkpoint is not None:
            checkpoint = load_checkpoint(segmentor, checkpoint, map_location='cpu')
            segmentor.CLASSES = checkpoint['meta']['CLASSES']
            segmentor.PALETTE = checkpoint['meta']['PALETTE']
        segmentor.cfg = config
    else:
        raise NotImplementedError
    segmentor.to(device)
    segmentor.eval()
    return segmentor

def fcn_r50_d8_drive(config, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    checkpoint = os.path.join(pretrained_root_path, 'fcn_r50-d8_512x1024_80k_cityscapes_20200606_113019-03aa804d.pth')
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

def fcn_r18_d8(config, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    checkpoint = os.path.join(pretrained_root_path, 'fcn_r18-d8_512x1024_80k_cityscapes_20201225_021327-6c50f8b4.pth')
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
    model_config = get_fcn_r18_d8_320x320_10k_cityscapes_config()
    model = fcn_r18_d8_320_320(model_config)
    
    print(model)
    print('model size {:.3f}MB'.format(get_model_size(model) / 1024**2))
    print('backbone size {:.3f}MB'.format(get_model_size(get_module(model, 'backbone')) / 1024**2))
    # print('neck size {:.3f}MB'.format(get_model_size(get_module(model, 'neck')) / 1024**2))
    print('decode_head size {:.3f}MB'.format(get_model_size(get_module(model, 'decode_head')) / 1024**2))
    print('auxiliary_head size {:.3f}MB'.format(get_model_size(get_module(model, 'auxiliary_head')) / 1024**2))
    exit(0)
    model = fcn_r18_d8('/data/gxy/legodnn-public-version_semantic_segmentation/cv_task/semantic_segmentation/mmseg_models/legodnn_configs/fcn_r18-d8_512x1024_80k_cityscapes.py')
    
    print(model)
    # print('model size {:.3f}MB'.format(get_model_size(model) / 1024**2))
    # print('backbone size {:.3f}MB'.format(get_model_size(get_module(model, 'backbone')) / 1024**2))
    # # print('neck size {:.3f}MB'.format(get_model_size(get_module(model, 'neck')) / 1024**2))
    # print('decode_head size {:.3f}MB'.format(get_model_size(get_module(model, 'decode_head')) / 1024**2))
    # print('auxiliary_head size {:.3f}MB'.format(get_model_size(get_module(model, 'auxiliary_head')) / 1024**2))


    model = fcn_r18_d8('/data/gxy/legodnn-public-version_semantic_segmentation/cv_task/semantic_segmentation/mmseg_models/legodnn_configs/fcn_r18-d8_512x1024_80k_cityscapes.py')
    print('model size {:.3f}MB'.format(get_model_size(model) / 1024**2))
    print(model)
    # input = torch.rand((1,3,224,224)).cuda()
    # model(input)