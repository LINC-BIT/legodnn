
from mmcv import Config
import sys
root_path = 'cv_task/object_detection/mmdet_models/legodnn_configs/'
    
faster_rcnn_r50_fpn_1x_coco_config = root_path + 'faster_rcnn_r50_fpn_1x_coco.py'
faster_rcnn_r50_fpn_1x_coco_fine_tuning_config = root_path + 'faster_rcnn_r50_fpn_1x_coco_fine_tuning.py'

yolov3_d53_320_273e_coco_config = root_path + 'yolov3_d53_320_273e_coco.py'
yolov3_d53_320_273e_coco_fine_tuning_config = root_path + 'yolov3_d53_320_273e_coco_fine_tuning.py'
yolov3_d53_320_273e_coco_usnet_config = root_path + 'yolov3_d53_320_273e_coco_usnet.py'

centernet_resnet18_dcnv2_140e_coco_config = root_path + 'centernet_resnet18_dcnv2_140e_coco.py'
fcos_x101_64x4d_fpn_gn_head_mstrain_640_800_2x_coco_config = root_path + 'fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py'
retinanet_free_anchor_r50_fpn_1x_coco_config = root_path + 'retinanet_free_anchor_r50_fpn_1x_coco.py'


def get_faster_rcnn_r50_fpn_1x_b16_voc2007_config(mode='train'):
    assert mode in ['train']
    if mode=='train':
        cfg = Config.fromfile(root_path + 'base_faster_rcnn_r50_fpn_1x_b16_voc2007.py')
    else:
        raise NotImplementedError
    return cfg

def get_faster_rcnn_r50_fpn_2x_b16_voc2007_config(mode='train'):
    assert mode in ['train', 'fn3', 'usnet', 'fine_tune', 'nestdnn', 'usnet_bn_cal']
    if mode=='train':
        cfg = Config.fromfile(root_path + 'faster_rcnn_r50_fpn_2x_b16_voc2007.py')
    elif mode=='fn3':
        cfg = Config.fromfile(root_path + 'faster_rcnn_r50_fpn_2x_b16_voc2007_fn3.py')
    elif mode=='usnet':
        cfg = Config.fromfile(root_path + 'faster_rcnn_r50_fpn_2x_b16_voc2007_usnet.py')
    elif mode=='usnet_bn_cal':
        cfg = Config.fromfile(root_path + 'faster_rcnn_r50_fpn_2x_b16_voc2007_usnet_bn_cal.py')
    elif mode=='nestdnn':
        cfg = Config.fromfile(root_path + 'faster_rcnn_r50_fpn_2x_b16_voc2007_nestdnn.py')
    elif mode=='fine_tune':
        cfg = Config.fromfile(root_path + 'faster_rcnn_r50_fpn_2x_b16_voc2007_fine_tune.py')
    else:
        raise NotImplementedError
    return cfg

def get_faster_rcnn_r50_fpn_4x_b16_voc2007_config(mode='train'):
    assert mode in ['train']
    if mode=='train':
        cfg = Config.fromfile(root_path + 'base_faster_rcnn_r50_fpn_4x_b16_voc2007.py')
    else:
        raise NotImplementedError
    return cfg

def get_yolov3_d53_320_160e_64b_voc07_config(mode='train'):
    assert mode in ['train', 'fn3', 'usnet', 'fine_tune', 'nestdnn', 'usnet_bn_cal']
    if mode=='train':
        cfg = Config.fromfile(root_path + 'yolov3_d53_320_160e_64b_voc07.py')
    elif mode=='usnet':
        cfg = Config.fromfile(root_path + 'yolov3_d53_320_160e_64b_voc07_usnet.py')
    elif mode=='usnet_bn_cal':
        cfg = Config.fromfile(root_path + 'yolov3_d53_320_160e_64b_voc07_usnet_bn_cal.py')
    elif mode=='fn3':
        cfg = Config.fromfile(root_path + 'yolov3_d53_320_160e_64b_voc07_fn3.py')
    elif mode=='fine_tune':
        cfg = Config.fromfile(root_path + 'yolov3_d53_320_160e_64b_voc07_fine_tune.py')
    elif mode=='nestdnn':
        cfg = Config.fromfile(root_path + 'yolov3_d53_320_160e_64b_voc07_nestdnn.py')
    else:
        raise NotImplementedError
    return cfg

def get_yolov3_d53_320_273e_64b_voc07_config(mode='train'):
    assert mode in ['train']
    if mode=='train':
        cfg = Config.fromfile(root_path + 'yolov3_d53_320_273e_64b_voc07.py')
    else:
        raise NotImplementedError
    return cfg
    
def get_faster_rcnn_r50_fpn_1x_coco_config(mode='train', input_size=None):
    assert mode in ['train', 'fine_tune']
    
    if mode=='train':
        cfg = Config.fromfile(faster_rcnn_r50_fpn_1x_coco_config)
    elif mode=='fine_tune':
        cfg = Config.fromfile(faster_rcnn_r50_fpn_1x_coco_fine_tuning_config)
    else:
        raise NotImplementedError
    
    if input_size is not None:
        cfg.data.legotrain.pipeline[1].img_scale = input_size[-2:]
        cfg.data.val.pipeline[1].img_scale = input_size[-2:]
        cfg.data.test.pipeline[1].img_scale = input_size[-2:]
    return cfg

def get_yolov3_d53_320_273e_coco_config(mode='train', input_size=None):
    # cfg = Config.fromfile(yolov3_d53_320_273e_coco_config)
    assert mode in ['train', 'fine_tune', 'usnet']
    
    if mode=='train':
        cfg = Config.fromfile(yolov3_d53_320_273e_coco_config)
    elif mode=='fine_tune':
        cfg = Config.fromfile(yolov3_d53_320_273e_coco_fine_tuning_config)
    elif mode=='usnet':
        cfg = Config.fromfile(yolov3_d53_320_273e_coco_usnet_config)
    else:
        raise NotImplementedError
    
    if input_size is not None:
        cfg.data.legotrain.pipeline[1].img_scale = input_size[-2:]
        cfg.data.val.pipeline[1].img_scale = input_size[-2:]
        cfg.data.test.pipeline[1].img_scale = input_size[-2:]
    return cfg

def get_retinanet_free_anchor_r50_fpn_1x_coco_config(input_size):
    cfg = Config.fromfile(retinanet_free_anchor_r50_fpn_1x_coco_config)
    cfg.data.legotrain.pipeline[1].img_scale = input_size[-2:]
    cfg.data.legotrain.pipeline[1].transforms[0].keep_ratio = False
    
    cfg.data.test.pipeline[1].img_scale = input_size[-2:]
    cfg.data.test.pipeline[1].transforms[0].keep_ratio = False
    return cfg


if __name__=='__main__':
    a = get_faster_rcnn_r50_fpn_1x_coco_config(input_size=(224,224))
    print(a.data)