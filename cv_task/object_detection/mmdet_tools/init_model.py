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

def mmdet_init_model(config, checkpoint=None, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
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
                detector.CLASSES = get_classes('coco')
        detector.cfg = config
    else:
        raise NotImplementedError
    detector = detector.to(device)
    detector.eval()
    return detector