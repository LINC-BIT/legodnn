_base_ = './fcn_r50-d8_512x1024_80k_cityscapes.py'

work_dir = './fcn_r18-d8_512x1024_80k_cityscapes'
seed = 0

model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))