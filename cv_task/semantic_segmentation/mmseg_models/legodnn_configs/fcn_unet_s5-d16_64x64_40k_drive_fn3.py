_base_ = './fcn_unet_s5-d16_64x64_40k_drive.py'

work_dir = './fcn_unet_s5-d16_64x64_40k_drive_usnet'
seed = 0

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=100000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, pre_eval=True)