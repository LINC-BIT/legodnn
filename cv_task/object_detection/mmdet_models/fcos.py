import os
import sys
sys.path.insert(0, '../../../')

from mmdet.apis import init_detector
from cv_task.object_detection.mmdet_models.load_mode import LOAD_MODE

configs_root_path= '/data/gxy/legodnn-public-version_object_detection/cv_task/object_detection/mmdet_models/configs/'
pretrained_root_path = '/data/gxy/pretrained_models/mmdetection/'

def focs_x101_64x4d_fpn(config, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    # config = os.path.join(configs_root_path, 'fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py')
    checkpoint = os.path.join(pretrained_root_path, 'fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco-ede514a8.pth')
    
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
    model = focs_x101_64x4d_fpn(mode='mmdet_test')
    print(model)

