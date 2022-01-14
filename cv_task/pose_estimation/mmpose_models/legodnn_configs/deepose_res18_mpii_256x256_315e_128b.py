_base_ = "./deepose_res18_mpii_256x256_210e_128b.py"

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[255, 300])
total_epochs = 315

work_dir = './base_deeppose_res50_coco_256x192_315e_128b'
seed = 0