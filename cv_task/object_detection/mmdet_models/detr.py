import os
import sys
sys.path.insert(0, '../../../')

from mmdet.apis import init_detector

configs_root_path= '/data/gxy/legodnn-public-version_object_detection/cv_task/object_detection/mmdet_models/configs/'
pretrained_root_path = '/data/gxy/pretrained_models/mmdetection/'

def detr_r50(device='cuda'):
    config = os.path.join(configs_root_path, 'detr/detr_r50_8x2_150e_coco.py')
    checkpoint = os.path.join(pretrained_root_path, 'detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth')
    detector = init_detector(config, checkpoint, device=device) 
    return detector


if __name__=='__main__':
    model = detr_r50()
    print(model)

