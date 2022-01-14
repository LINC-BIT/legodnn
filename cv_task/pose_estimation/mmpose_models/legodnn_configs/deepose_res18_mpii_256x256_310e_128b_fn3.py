_base_ = "./deepose_res18_mpii_256x256_310e_128b.py"

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
    step=[600, 725])
total_epochs = 775

work_dir = './base_deeppose_res50_coco_256x192_310e_128b_fn3'
seed = 0