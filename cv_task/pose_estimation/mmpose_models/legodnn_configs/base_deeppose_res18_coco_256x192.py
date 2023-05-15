_base_ = './base_deeppose_res50_coco_256x192.py'

# model settings
model = dict(
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', 
                  depth=18),
    keypoint_head=dict(
        in_channels=512),
    )

work_dir = './base_deeppose_res50_coco_256x192'
seed = 0
