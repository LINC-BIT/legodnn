_base_ = './fcn_r18-d8_512x512_b16_30k_voc2012_fine_tune.py'

work_dir = './fcn_r18-d8_512x512_b16_30k_voc2012_nestdnn'
seed = 0

optimizer = dict(weight_decay=0.0)