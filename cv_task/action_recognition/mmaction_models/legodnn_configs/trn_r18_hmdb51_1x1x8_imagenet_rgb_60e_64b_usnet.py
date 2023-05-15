_base_ = ['./base_trn_r18_hmdb51_1x1x8_imagenet_rgb.py']


work_dir = './trn_r18_hmdb51_1x1x8_imagenet_rgb_60e_64b_usnet'
seed = 0


# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001)

# learning policy
lr_config = dict(policy='step', step=[60, 75])
total_epochs = 90
