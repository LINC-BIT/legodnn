_base_ = './yolov3_d53_320_273e_coco.py'

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[109, 123])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=136)

method='usnet'