_base_ = './base_yolov3_d53_320_273e_voc2007.py'

work_dir = './yolov3_d53_320_160e_64b_voc07_fine_tune'
seed = 0

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=400,  # same as burn-in in darknet
    warmup_ratio=0.001,
    step=[16])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=32)
# evaluation = dict(interval=10, metric=['mAP'])
evaluation = dict(interval=10, metric='mAP')
