from mmcv import Config

cfg = Config.fromfile('cv_task/object_detection/mmdet_models/configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py')
print(cfg.lr_config)