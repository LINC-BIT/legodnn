_base_ = './fcn_r18-d8_512x512_b16_30k_voc2012.py'

work_dir = './fcn_r18-d8_512x512_b16_30k_voc2012_test'
seed = 0

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=600)
checkpoint_config = dict(by_epoch=False, interval=60)
evaluation = dict(interval=600, pre_eval=True)

workflow = [('train', 1)]