import os
import sys

import torch
sys.path.insert(0, '../../../')

from mmpose.apis import init_pose_model
from cv_task.pose_estimation.mmpose_models.load_mode import LOAD_MODE
from legodnn.utils.dl.common.model import get_model_size, get_module
from cv_task.pose_estimation.mmpose_models.legodnn_configs import get_simplebaseline_res50_coco_256x192_config
pretrained_root_path = '/data/gxy/pretrained_models/mmpose/'

def simplebaseline_res50_coco_256x192(config, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    checkpoint = os.path.join(pretrained_root_path, 'simplebaseline_res50_coco_256x192-ec54d7f3_20200709.pth')
    if mode=='lego_jit':
        pose_model = init_pose_model(config, checkpoint, device=device)
        pose_model.forward = pose_model.forward_dummy
        
    elif mode=='mmpose_test':
        pose_model = init_pose_model(config, checkpoint, device=device)
        pose_model = pose_model
    else:
        raise NotImplementedError
    pose_model.eval()
    return pose_model

if __name__=='__main__':
    model_config = get_simplebaseline_res50_coco_256x192_config()
    model = simplebaseline_res50_coco_256x192(model_config)

    print(model)
    print('model size {:.3f}MB'.format(get_model_size(model) / 1024**2))
    print('backbone size {:.3f}MB'.format(get_model_size(get_module(model, 'backbone')) / 1024**2))
    # print('neck size {:.3f}MB'.format(get_model_size(get_module(model, 'neck')) / 1024**2))
    print('keypoint_head size {:.3f}MB'.format(get_model_size(get_module(model, 'keypoint_head')) / 1024**2))