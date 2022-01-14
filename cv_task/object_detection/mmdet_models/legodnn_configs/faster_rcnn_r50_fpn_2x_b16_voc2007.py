_base_ = [
    './base_faster_rcnn_r50_fpn_2x_b16_voc2007.py'
]

lr_config = dict(policy='step', 
                warmup='linear',
                warmup_iters=500,
                warmup_ratio=0.001)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4
    )

work_dir = './faster_rcnn_r50_2x_b16_voc_raw'
seed = 0