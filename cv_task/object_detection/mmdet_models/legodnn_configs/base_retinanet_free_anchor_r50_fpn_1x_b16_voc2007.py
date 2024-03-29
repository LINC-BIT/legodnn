_base_ = ['../configs/_base_/models/retinanet_r50_fpn.py',
          '../configs/_base_/schedules/schedule_1x.py', 
          '../configs/_base_/datasets/voc2007.py',
          '../configs/_base_/default_runtime.py',
          ]
model = dict(bbox_head=dict(num_classes=20))
# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# # learning policy
# # actual epoch = 3 * 3 = 9
# lr_config = dict(policy='step', 
#                 # warmup='linear',
#                 # warmup_iters=100,
#                 # warmup_ratio=0.001,
#                 step=[6])
# runner = dict(
#     type='EpochBasedRunner', max_epochs=8)  # actual epoch = 4 * 3 = 12

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4
    )

workflow = [('train', 1), ('val', 1)]


work_dir = './faster_rcnn_r50_2x_b16_voc_raw'
seed = 0
