_base_ = "./base_deepose_res18_mpii_256x256.py"

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
)

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[240, 290])
total_epochs = 310
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

work_dir = './base_deeppose_res50_coco_256x192_310e_128b'
seed = 0