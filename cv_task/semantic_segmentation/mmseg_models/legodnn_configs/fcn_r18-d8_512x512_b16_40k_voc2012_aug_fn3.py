_base_ = './fcn_r18-d8_512x512_b16_40k_voc2012_aug.py'

work_dir = './fcn_r18-d8_512x512_b16_40k_voc2012_aug_fn3'
seed = 0

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=100000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, pre_eval=True)