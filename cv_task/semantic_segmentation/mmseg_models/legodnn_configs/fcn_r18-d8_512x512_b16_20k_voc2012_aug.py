_base_ = [
    './base_models_fcn_r18-d8.py',
    './base_datasets_pascal_voc12_aug.py',
    '../configs/_base_/default_runtime.py', 
    '../configs/_base_/schedules/schedule_20k.py'
]

work_dir = './fcn_r18-d8_512x512_b16_20k_voc2012_aug'
seed = 0