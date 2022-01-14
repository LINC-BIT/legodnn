_base_ = './fcn_r18-d8_512x512_b16_20k_voc2012_aug.py'

work_dir = './fcn_r18-d8_512x512_b16_20k_voc2012_aug_usnet'
seed = 0

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=30000)
checkpoint_config = dict(by_epoch=False, interval=3000)
evaluation = dict(interval=3000, pre_eval=True)