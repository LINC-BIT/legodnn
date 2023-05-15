_base_ = "./deepose_res18_mpii_256x256_310e_128b.py"

optimizer = dict(
    type='Adam',
    # lr=5e-5,
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=0.001,
    step=[49, 59])
total_epochs = 62

work_dir = './base_deeppose_res50_coco_256x192_310e_128b_fine_tune'
seed = 0