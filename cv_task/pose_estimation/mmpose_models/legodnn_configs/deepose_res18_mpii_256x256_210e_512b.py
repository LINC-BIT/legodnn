_base_ = "./base_deepose_res18_mpii_256x256.py"

# model settings
model = dict(
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18, num_stages=4, out_indices=(3, )),
    keypoint_head=dict(
        in_channels=512)
    )

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

work_dir = './base_deeppose_res50_coco_256x192'
seed = 0