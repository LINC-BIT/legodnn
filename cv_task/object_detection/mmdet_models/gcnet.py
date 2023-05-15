import os
import sys
sys.path.insert(0, '../../../')

from mmdet.apis import init_detector

configs_root_path= '/data/gxy/legodnn-public-version_object_detection/cv_task/object_detection/mmdet_models/configs/'
pretrained_root_path = '/data/gxy/pretrained_models/mmdetection/'

def gcnet_r50_fpn(device='cuda'):
    config = os.path.join(configs_root_path, 'gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py')
    checkpoint = os.path.join(pretrained_root_path, 'mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth')
    detector = init_detector(config, checkpoint, device=device) 
    return detector


if __name__=='__main__':
    model = gcnet_r50_fpn()
    print(model)

