import os
import sys
sys.path.insert(0, '../../../')

from mmdet.apis import init_detector
from cv_task.object_detection.mmdet_models.load_mode import LOAD_MODE
from cv_task.object_detection.mmdet_models.legodnn_configs import get_retinanet_free_anchor_r50_fpn_1x_coco_config
from legodnn.utils.dl.common.model import get_model_size, get_module

pretrained_root_path = '/data/gxy/pretrained_models/mmdetection/'

def free_anchor_r50_fpn(config, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    checkpoint = os.path.join(pretrained_root_path, 'retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth')
    
    if mode=='lego_jit':
        detector = init_detector(config, checkpoint, device=device)
        detector.forward = detector.forward_dummy
        
    elif mode=='mmdet_test':
        detector = init_detector(config, checkpoint, device=device)
        detector = detector
    else:
        raise NotImplementedError
    detector.eval()
    return detector


if __name__=='__main__':
    model_config = get_retinanet_free_anchor_r50_fpn_1x_coco_config((1,3,224,224))
    model = free_anchor_r50_fpn(model_config, mode='mmdet_test')
    print(model)
    print('model size {:.3f}MB'.format(get_model_size(model) / 1024**2))
    print('backbone size {:.3f}MB'.format(get_model_size(get_module(model, 'backbone')) / 1024**2))
    print('neck size {:.3f}MB'.format(get_model_size(get_module(model, 'neck')) / 1024**2))
    print('bbox_head size {:.3f}MB'.format(get_model_size(get_module(model, 'bbox_head')) / 1024**2))
