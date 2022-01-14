import os
import sys
sys.path.insert(0, '../../../')

from mmdet.apis import init_detector
from legodnn.utils.dl.common.model import get_model_size, get_module
from cv_task.object_detection.mmdet_models.load_mode import LOAD_MODE

configs_root_path= '/data/gxy/legodnn-public-version_object_detection/cv_task/object_detection/mmdet_models/configs/'
pretrained_root_path = '/data/gxy/pretrained_models/mmdetection/'

def double_head_rcnn_r50_fpn(mode='', device='cuda'):
    assert mode in LOAD_MODE
    config = os.path.join(configs_root_path, 'double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py')
    checkpoint = os.path.join(pretrained_root_path, 'dh_faster_rcnn_r50_fpn_1x_coco_20200130-586b67df.pth')
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
    model = double_head_rcnn_r50_fpn()
    print(model)
    print(get_model_size(model))
    print(get_model_size(get_module(model, 'roi_head.bbox_head')))

