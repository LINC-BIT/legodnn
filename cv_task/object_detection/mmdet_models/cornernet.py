import os
import sys
sys.path.insert(0, '../../../')

from mmdet.apis import init_detector

configs_root_path= '/data/gxy/legodnn-public-version_object_detection/cv_task/object_detection/mmdet_models/configs/'
pretrained_root_path = '/data/gxy/pretrained_models/mmdetection/'

def cornernet_r50_fpn(device='cuda'):
    config = os.path.join(configs_root_path, 'cornernet/cornernet_hourglass104_mstest_10x5_210e_coco.py')
    checkpoint = os.path.join(pretrained_root_path, 'cornernet_hourglass104_mstest_10x5_210e_coco_20200824_185720-5fefbf1c.pth')
    detector = init_detector(config, checkpoint, device=device) 
    return detector


if __name__=='__main__':
    model = cornernet_r50_fpn()
    print(model)

