_base_ = [
    './faster_rcnn_r50_fpn_2x_b16_voc2007.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
# lr_config = dict(policy='step', step=[9])
lr_config = dict(policy='step', 
                warmup='linear',
                warmup_iters=100,
                warmup_ratio=0.001,
                step=[9])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=12)  # actual epoch = 4 * 3 = 12

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4
    )


workflow = [('train', 1), ('val', 1)]


work_dir = './faster_rcnn_r50_2x_b16_voc_usnet'
seed = 0