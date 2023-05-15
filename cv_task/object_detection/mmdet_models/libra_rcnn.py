import os
import sys
sys.path.insert(0, '../../../')

from mmdet.apis import init_detector
from cv_task.object_detection.mmdet_models.load_mode import LOAD_MODE
configs_root_path= '/data/gxy/legodnn-public-version_object_detection/cv_task/object_detection/mmdet_models/configs/'
pretrained_root_path = '/data/gxy/pretrained_models/mmdetection/'

def libra_rcnn_r50_fpn(device='cuda'):
    config = os.path.join(configs_root_path, 'libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py')
    checkpoint = os.path.join(pretrained_root_path, 'libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth')
    detector = init_detector(config, checkpoint, device=device) 
    return detector


if __name__=='__main__':
    model = libra_rcnn_r50_fpn()
    print(model)

