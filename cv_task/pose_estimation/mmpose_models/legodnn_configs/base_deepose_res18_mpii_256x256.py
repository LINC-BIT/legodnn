_base_ = './base_deepose_res50_mpii_256x256.py'

# model settings
model = dict(
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', 
                  depth=18),
    keypoint_head=dict(
        in_channels=512),
    )

work_dir = './base_deepose_res50_mpii_256x256'
seed = 0
