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
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[50, 59])
total_epochs = 62

work_dir = './simplebaseline_res18_mpii_256x256_fine_tune.py'
seed = 0