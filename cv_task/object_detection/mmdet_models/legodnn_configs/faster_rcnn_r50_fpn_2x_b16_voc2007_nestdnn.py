_base_ = [
    './faster_rcnn_r50_fpn_2x_b16_voc2007_fine_tune.py'
]

optimizer = dict(weight_decay=0.0)

work_dir = './faster_rcnn_r50_2x_b16_voc_nestdnn'
seed = 0