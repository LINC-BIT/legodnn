_base_ = './deeplabv3_r18-d8_512x512_b16_30k_voc2012.py'

work_dir = './deeplabv3_r18-d8_512x512_b16_30k_voc2012_usnet'
seed = 0

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=45000)
checkpoint_config = dict(by_epoch=False, interval=4500)
evaluation = dict(interval=4500, pre_eval=True)
