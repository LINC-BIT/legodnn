_base_ = './fcn_r18-d8_320x320_10k_cityscapes_1210.py'

work_dir = './fcn_r18-d8_320x320_20k_cityscapes_usnet'
seed = 0

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=15000)
checkpoint_config = dict(by_epoch=False, interval=1500)
evaluation = dict(interval=1500, metric='mIoU', pre_eval=True)

