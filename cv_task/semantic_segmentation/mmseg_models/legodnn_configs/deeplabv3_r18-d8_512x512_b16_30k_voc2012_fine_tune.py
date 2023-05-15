_base_ = './deeplabv3_r18-d8_512x512_b16_30k_voc2012.py'

work_dir = './deeplabv3_r18-d8_512x512_b16_30k_voc2012_fine_tune'
seed = 0

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=6000)
checkpoint_config = dict(by_epoch=False, interval=600)
evaluation = dict(interval=600, pre_eval=True)
