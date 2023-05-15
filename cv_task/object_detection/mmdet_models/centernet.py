import os
import sys
sys.path.insert(0, '../../../')

from mmdet.apis import init_detector
from cv_task.object_detection.mmdet_models.load_mode import LOAD_MODE

configs_root_path= '/data/gxy/legodnn-public-version_object_detection/cv_task/object_detection/mmdet_models/configs/'
pretrained_root_path = '/data/gxy/pretrained_models/mmdetection/'

def centernet_resnet18_dcnv2_140e(config, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    # config = os.path.join(configs_root_path, 'centernet/centernet_resnet18_dcnv2_140e_coco.py')
    checkpoint = os.path.join(pretrained_root_path, 'centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth')
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
    model = centernet_resnet18_dcnv2_140e()
    print(model)
    # print(get_model_size(get_module(model, 'bbox_head')))

