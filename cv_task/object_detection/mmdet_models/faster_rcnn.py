import os
import sys
import warnings
sys.path.insert(0, '../../../')

import torch
from mmdet.apis import init_detector
from legodnn.utils.dl.common.model import get_model_size, get_module
from cv_task.object_detection.mmdet_models.load_mode import LOAD_MODE
from legodnn.utils.dl.common.model import save_model, ModelSaveMethod
from cv_task.object_detection.mmdet_models.legodnn_configs import get_faster_rcnn_r50_fpn_1x_coco_config

from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes

pretrained_root_path = '/data/gxy/pretrained_models/mmdetection/'

def faster_rcnn_r50_fpn(config, pretrained: bool, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    if pretrained:
        checkpoint = os.path.join(pretrained_root_path, 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')
    else:
        checkpoint = None

    if mode=='lego_jit':
        detector = init_detector(config, checkpoint, device=device)
        detector.forward = detector.forward_dummy
        detector.eval()
    elif mode=='mmdet_test':
        detector = init_detector(config, checkpoint, device=device)
        detector = detector
        detector.eval()
    elif mode=='mmdet_train':
        
        detector = build_detector(config.model, train_cfg=config.get('train_cfg'), test_cfg=config.get('test_cfg'))
        detector.init_weights()
        if checkpoint is not None:
            map_loc = 'cpu' if device == 'cpu' else None
            checkpoint = load_checkpoint(detector, checkpoint, map_location=map_loc)
            if 'CLASSES' in checkpoint.get('meta', {}):
                detector.CLASSES = checkpoint['meta']['CLASSES']
            else:
                warnings.simplefilter('once')
                warnings.warn('Class names are not saved in the checkpoint\'s '
                            'meta data, use COCO classes by default.')
                model.CLASSES = get_classes('coco')
                
        detector = detector
    else:
        raise NotImplementedError
    
    return detector


if __name__=='__main__':
    model_config = get_faster_rcnn_r50_fpn_1x_coco_config((1,3,224,224))
    model = faster_rcnn_r50_fpn(model_config, mode='mmdet_test')
    # save_model(model, '/data/gxy/legodnn-public-version_object_detection/cv_task/object_detection/mmdet_models/save_models/faster_rcnn_r50_fpn.pt', ModelSaveMethod.FULL)
    print(model)
    print('model size {:.3f}MB'.format(get_model_size(model) / 1024**2))
    print('backbone size {:.3f}MB'.format(get_model_size(get_module(model, 'backbone')) / 1024**2))
    print('neck size {:.3f}MB'.format(get_model_size(get_module(model, 'neck')) / 1024**2))
    print('rpn_head size {:.3f}MB'.format(get_model_size(get_module(model, 'rpn_head')) / 1024**2))
    print('roi_head size {:.3f}MB'.format(get_model_size(get_module(model, 'roi_head')) / 1024**2))
    
    print('roi_head size {:.3f}MB'.format(get_model_size(get_module(model, 'roi_head')) / 1024**2))
    print('roi_head.bbox_roi_extractor size {:.3f}MB'.format(get_model_size(get_module(model, 'roi_head.bbox_roi_extractor')) / 1024**2))
    print('roi_head.bbox_head size {:.3f}MB'.format(get_model_size(get_module(model, 'roi_head.bbox_head')) / 1024**2))
    
    print('rpn_head size {:.3f}MB'.format(get_model_size(get_module(model, 'rpn_head')) / 1024**2))
    # print(model)

