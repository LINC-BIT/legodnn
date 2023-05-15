_base_ = [
    '../configs/_base_/models/faster_rcnn_r50_fpn.py', '../configs/_base_/datasets/voc2007.py',
    '../configs/_base_/default_runtime.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=4)  # actual epoch = 4 * 3 = 12

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4
    )
workflow = [('train', 1), ('val', 1)]

work_dir = './faster_rcnn_r50_1x_b16_voc_raw'
seed = 0