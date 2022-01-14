_base_ = [
    './base_models_fcn_r18-d8.py',
    './base_datasets_pascal_voc12.py',
    '../configs/_base_/default_runtime.py', 
    '../configs/_base_/schedules/schedule_30k.py'
]

work_dir = './fcn_r18-d8_512x512_b16_30k_voc2012'
seed = 0
