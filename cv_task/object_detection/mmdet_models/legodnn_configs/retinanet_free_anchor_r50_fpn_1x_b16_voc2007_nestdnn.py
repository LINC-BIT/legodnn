_base_ = ['retinanet_free_anchor_r50_fpn_1x_b16_voc2007_fine_tune.py']

optimizer = dict(weight_decay=0.0)


work_dir = './retinanet_free_anchor_r50_fpn_1x_b16_voc2007_raw'
seed = 0
