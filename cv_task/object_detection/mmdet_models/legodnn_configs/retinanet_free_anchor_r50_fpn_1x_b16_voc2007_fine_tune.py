_base_ = ['retinanet_free_anchor_r50_fpn_1x_b16_voc2007.py']

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[])
runner = dict(type='EpochBasedRunner', max_epochs=2)


work_dir = './retinanet_free_anchor_r50_fpn_1x_b16_voc2007_fine_tune'
seed = 0
