_base_ = "./simplebaseline_res18_mpii_256x256_310e_128b.py"

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
    step=[625, 737])
total_epochs = 775

work_dir = './simplebaseline_res18_mpii_256x256_fn3.py'
seed = 0