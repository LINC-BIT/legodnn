import os
import sys
import warnings
sys.path.insert(0, '../../../')

from mmdet.apis import init_detector
from cv_task.object_detection.mmdet_models.load_mode import LOAD_MODE
from legodnn.utils.dl.common.model import get_model_size, get_module
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from cv_task.object_detection.mmdet_models.legodnn_configs import get_yolov3_d53_320_273e_coco_config
pretrained_root_path = '/data/gxy/pretrained_models/mmdetection/'

def yolov3_darknet53(config, pretrained: bool, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    if pretrained:
        checkpoint = os.path.join(pretrained_root_path, 'yolov3_d53_320_273e_coco-421362b6.pth')
    else:
        checkpoint = None

    if mode=='lego_jit':
        detector = init_detector(config, checkpoint, device=device)
        detector.forward = detector.forward_dummy
        
    elif mode=='mmdet_test':
        detector = init_detector(config, checkpoint, device=device)
        detector = detector
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
    detector.eval()
    return detector

if __name__=='__main__':
    model_config = get_yolov3_d53_320_273e_coco_config()
    model = yolov3_darknet53(model_config, pretrained=False)
    print(model)
    print('model size {:.3f}MB'.format(get_model_size(model) / 1024**2))
    print('backbone size {:.3f}MB'.format(get_model_size(get_module(model, 'backbone')) / 1024**2))
    print('neck size {:.3f}MB'.format(get_model_size(get_module(model, 'neck')) / 1024**2))
    print('bbox_head size {:.3f}MB'.format(get_model_size(get_module(model, 'bbox_head')) / 1024**2))


