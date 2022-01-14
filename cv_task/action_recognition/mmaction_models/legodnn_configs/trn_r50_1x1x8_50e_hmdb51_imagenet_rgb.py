_base_ = [
    '../configs/_base_/models/trn_r50.py', 
    '../configs/_base_/schedules/sgd_50e.py',
    '../configs/_base_/default_runtime.py'
]

# model settings
model = dict(cls_head=dict(num_classes=51))

work_dir = './work_dirs/tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb'

# dataset settings
split = 1
dataset_type = 'RawframeDataset'
data_root = '/data/datasets/hmdb51/rawframes'
data_root_val = '/data/datasets/hmdb51/rawframes'
ann_file_train = f'/data/datasets/hmdb51/hmdb51_train_split_{split}_rawframes.txt'
ann_file_val = f'/data/datasets/hmdb51/hmdb51_val_split_{split}_rawframes.txt'
ann_file_test = f'/data/datasets/hmdb51/hmdb51_val_split_{split}_rawframes.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

legodnn_train_pipeline = test_pipeline

data = dict(
    # videos_per_gpu=64,
    # workers_per_gpu=1,
    # test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    legotrain=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=legodnn_train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))

# optimizer
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/trn_r50_1x1x8_50e_hmdb51_imagenet_rgb/'
gpu_ids = range(0, 1)
