import os
import sys
sys.path.insert(0, '../../../')

from mmseg.apis import init_segmentor
from cv_task.semantic_segmentation.mmseg_models.load_mode import LOAD_MODE
from legodnn.utils.dl.common.model import get_model_size, get_module
from cv_task.semantic_segmentation.mmseg_models.legodnn_configs import get_emanet_r50_d8_512x1024_80k_cityscapes_config

pretrained_root_path = '/data/gxy/pretrained_models/mmsegmentation/'

def emanet_r50_d8(config, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    checkpoint = os.path.join(pretrained_root_path, 'emanet_r50-d8_512x1024_80k_cityscapes_20200901_100301-c43fcef1.pth')
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
    model_config = get_emanet_r50_d8_512x1024_80k_cityscapes_config((1,3,320,320))
    model = emanet_r50_d8(model_config)

    print(model)
    print('model size {:.3f}MB'.format(get_model_size(model) / 1024**2))
    print('backbone size {:.3f}MB'.format(get_model_size(get_module(model, 'backbone')) / 1024**2))
    # print('neck size {:.3f}MB'.format(get_model_size(get_module(model, 'neck')) / 1024**2))
    print('decode_head size {:.3f}MB'.format(get_model_size(get_module(model, 'decode_head')) / 1024**2))
    print('auxiliary_head size {:.3f}MB'.format(get_model_size(get_module(model, 'auxiliary_head')) / 1024**2))


