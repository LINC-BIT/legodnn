_base_ = './base_models_fcn_r50-d8.py'

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
        num_classes=21,
    ),
    auxiliary_head=dict(
        in_channels=256,
        channels=64,
        num_classes=21,
        )
    )