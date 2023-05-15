_base_ = './yolov3_d53_320_160e_64b_voc07_fine_tune.py'

work_dir = './yolov3_d53_320_160e_64b_voc07_nestdnn'
seed = 0

# optimizer
optimizer = dict(weight_decay=0.0)
