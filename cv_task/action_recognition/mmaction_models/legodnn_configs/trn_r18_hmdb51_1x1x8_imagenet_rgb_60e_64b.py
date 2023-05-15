_base_ = ['./base_trn_r18_hmdb51_1x1x8_imagenet_rgb.py']


work_dir = './trn_r18_hmdb51_1x1x8_imagenet_rgb_60e_64b'
seed = 0


data = dict(
    videos_per_gpu=64,
    workers_per_gpu=4
)


# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[40, 50])
total_epochs = 60
