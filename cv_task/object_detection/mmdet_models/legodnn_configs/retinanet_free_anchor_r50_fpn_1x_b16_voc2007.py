_base_ = ['base_retinanet_free_anchor_r50_fpn_1x_b16_voc2007.py']

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4
    )

workflow = [('train', 1), ('val', 1)]


work_dir = './retinanet_free_anchor_r50_fpn_1x_b16_voc2007_raw'
seed = 0
