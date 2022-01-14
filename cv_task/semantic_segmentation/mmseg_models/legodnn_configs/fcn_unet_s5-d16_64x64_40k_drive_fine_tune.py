_base_ = './fcn_unet_s5-d16_64x64_40k_drive.py'

work_dir = './fcn_unet_s5-d16_64x64_40k_drive_fine_tune'
seed = 0

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=8000)
checkpoint_config = dict(by_epoch=False, interval=800)
evaluation = dict(interval=800, pre_eval=True)

