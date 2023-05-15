_base_ = "./base_simplebaseline_res18_mpii_256x256.py"

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
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
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[250, 295])
total_epochs = 310
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

work_dir = './simplebaseline_res18_mpii_256x256.py'
seed = 0