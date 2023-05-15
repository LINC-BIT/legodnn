import os
import sys
sys.path.insert(0, '../../../')
from mmdet.apis import init_detector

configs_root_path= '/data/gxy/legodnn-public-version_object_detection/cv_task/object_detection/mmdet_models/configs/'
pretrained_root_path = '/data/gxy/pretrained_models/mmdetection/'

def mask_rcnn_r50_fpn(device='cuda'):
    config = os.path.join(configs_root_path, 'mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py')
    checkpoint = os.path.join(pretrained_root_path, 'mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth')
    detector = init_detector(config, checkpoint, device=device) 
    return detector

if __name__=='__main__':
    model = mask_rcnn_r50_fpn()
    print(model)