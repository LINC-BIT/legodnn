_base_ = ['./trn_r50_1x1x8_50e_hmdb51_imagenet_rgb.py']

# model settings
model = dict(
    backbone=dict(
        type='ResNet', 
        pretrained='torchvision://resnet18', 
        depth=18),
    cls_head=dict(
        in_channels=512),
    )


work_dir = './base_trn_r18_hmdb51_1x1x8_imagenet_rgb'
seed = 0


data = dict(
    videos_per_gpu=64,
    workers_per_gpu=4
    )


# # optimizer
# optimizer = dict(
#     type='SGD',
#     lr=0.01,  # this lr is used for 8 gpus
#     momentum=0.9,
#     weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[20, 30])
total_epochs = 35
