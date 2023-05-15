_base_ = ['./tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_fine_tune.py']

work_dir = './tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_nestdnn'

optimizer = dict(weight_decay=0.0)
