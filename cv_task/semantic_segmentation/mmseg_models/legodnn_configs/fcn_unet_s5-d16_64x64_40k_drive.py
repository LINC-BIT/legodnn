_base_ = [
    './base_models_fcn_unet_s5-d16.py', './base_datasets_64_64_4b_drive.py',
    '../configs/_base_/default_runtime.py', '../configs/_base_/schedules/schedule_40k.py'
]

model = dict(test_cfg=dict(crop_size=(64, 64), stride=(42, 42)))
evaluation = dict(metric='mDice')

work_dir = './fcn_unet_s5-d16_64x64_40k_drive'
seed = 0