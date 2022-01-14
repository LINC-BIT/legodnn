from mmcv import Config
import sys
# sys.path.insert(0, '../../../../')
root_path = 'cv_task/semantic_segmentation/mmseg_models/legodnn_configs/'


fcn_r50_d8_512x1024_80k_cityscapes_config = root_path + 'fcn_r50-d8_512x1024_80k_cityscapes.py'
fcn_r18_d8_512x1024_80k_cityscapes_config = root_path + 'fcn_r18-d8_512x1024_80k_cityscapes.py'

fcn_r18_d8_320x320_10k_cityscapes_config = root_path + 'fcn_r18-d8_320x320_10k_cityscapes_1210.py'
fcn_r18_d8_320x320_10k_cityscapes_fine_tune_config = root_path + 'fcn_r18-d8_320x320_10k_cityscapes_fine_tune.py'
fcn_r18_d8_320x320_10k_cityscapes_fn3_config = root_path + 'fcn_r18-d8_320x320_10k_cityscapes_fn3.py'
fcn_r18_d8_320x320_10k_cityscapes_usnet_config = root_path + 'fcn_r18-d8_320x320_10k_cityscapes_usnet.py'
fcn_r18_d8_320x320_10k_cityscapes_cgnet_config = root_path + 'fcn_r18-d8_320x320_10k_cityscapes_cgnet.py'

deeplabv3_r18_d8_512x1024_80k_cityscapes_config = root_path + 'deeplabv3_r18-d8_512x1024_80k_cityscapes.py'
ccnet_r50_d8_512x1024_80k_cityscapes_config = root_path + 'ccnet_r50-d8_512x1024_80k_cityscapes.py'
fcn_unet_s5_d16_64x64_40k_drive_config = root_path + 'fcn_unet_s5-d16_64x64_40k_drive.py'
gcnet_r50_d8_512x1024_40k_cityscapes_config = root_path + 'gcnet_r50-d8_512x1024_40k_cityscapes.py'
emanet_r50_d8_512x1024_80k_cityscapes_config = root_path + 'emanet_r50-d8_512x1024_80k_cityscapes.py'
pointrend_r50_512x1024_80k_cityscapes_config = root_path + 'pointrend_r50_512x1024_80k_cityscapes.py'
dmnet_r50_d8_512x1024_80k_cityscapes_config = root_path + 'dmnet_r50-d8_512x1024_80k_cityscapes.py'


fcn_r18_d8_320x320_10k_voc2012_config = root_path + 'fcn_r18-d8_320x320_10k_voc2012.py'
fcn_r18_d8_320x320_20k_voc2012_config = root_path + 'fcn_r18-d8_320x320_20k_voc2012.py'
fcn_r18_d8_320x320_30k_voc2012_config = root_path + 'fcn_r18-d8_320x320_30k_voc2012.py'
fcn_r18_d8_320x320_40k_voc2012_config = root_path + 'fcn_r18-d8_320x320_40k_voc2012.py'


def get_fcn_unet_s5_d16_64x64_4b_40k_drive_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet']
    if mode=='train':
        cfg = Config.fromfile(root_path + 'fcn_unet_s5-d16_64x64_40k_drive.py')
    elif mode == 'usnet':
        cfg = Config.fromfile(root_path + 'fcn_unet_s5-d16_64x64_40k_drive_usnet.py')
    elif mode == 'fn3':
        cfg = Config.fromfile(root_path + 'fcn_unet_s5-d16_64x64_40k_drive_fn3.py')
    elif mode=='fine_tune':
        cfg = Config.fromfile(root_path + 'fcn_unet_s5-d16_64x64_40k_drive_fine_tune.py')
    else:
        raise NotImplementedError
    return cfg

def get_fcn_r18_d8_512x512_b16_40k_voc2012_aug_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet']
    
    if mode=='train':
        cfg = Config.fromfile(root_path + 'fcn_r18-d8_512x512_b16_40k_voc2012_aug.py')
    elif mode == 'usnet':
        cfg = Config.fromfile(root_path + 'fcn_r18-d8_512x512_b16_40k_voc2012_aug_usnet.py')
    elif mode == 'fn3':
        cfg = Config.fromfile(root_path + 'fcn_r18-d8_512x512_b16_40k_voc2012_aug_fn3.py')
    else:
        raise NotImplementedError
    return cfg

def get_fcn_r18_d8_512x512_b16_20k_voc2012_aug_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet']
    if mode=='train':
        cfg = Config.fromfile(root_path + 'fcn_r18-d8_512x512_b16_20k_voc2012_aug.py')
    elif mode == 'usnet':
        cfg = Config.fromfile(root_path + 'fcn_r18-d8_512x512_b16_20k_voc2012_aug_usnet.py')
    elif mode == 'fn3':
        cfg = Config.fromfile(root_path + 'fcn_r18-d8_512x512_b16_20k_voc2012_aug_fn3.py')
    else:
        raise NotImplementedError
    return cfg

def get_fcn_r18_d8_512x512_b16_30k_voc2012_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet', 'nestdnn', 'test', 'test_nestdnn']
    if mode in ['train', 'cgnet']:
        cfg = Config.fromfile(root_path + 'fcn_r18-d8_512x512_b16_30k_voc2012.py')
    elif mode == 'usnet':
        cfg = Config.fromfile(root_path + 'fcn_r18-d8_512x512_b16_30k_voc2012_usnet.py')
    elif mode == 'fn3':
        cfg = Config.fromfile(root_path + 'fcn_r18-d8_512x512_b16_30k_voc2012_fn3.py')
    elif mode=='fine_tune':
        cfg = Config.fromfile(root_path + 'fcn_r18-d8_512x512_b16_30k_voc2012_fine_tune.py')
    elif mode=='nestdnn':
        cfg = Config.fromfile(root_path + 'fcn_r18-d8_512x512_b16_30k_voc2012_nestdnn.py')
    elif mode=='test':
        cfg = Config.fromfile(root_path + 'fcn_r18-d8_512x512_b16_30k_voc2012_test.py')
    elif mode=='test_nestdnn':
        cfg = Config.fromfile(root_path + 'fcn_r18-d8_512x512_b16_30k_voc2012_test_nestdnn.py')
    else:
        raise NotImplementedError
    return cfg

fcn_unet_s5_d16_64x64_60k_drive_config = root_path + 'fcn_unet_s5-d16_64x64_60k_drive.py'
def get_fcn_unet_s5_d16_64x64_4b_60k_drive_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet']
    if mode=='train':
        cfg = Config.fromfile(fcn_unet_s5_d16_64x64_60k_drive_config)
    else:
        raise NotImplementedError
    return cfg

fcn_unet_s5_d16_64x64_80k_drive_config = root_path + 'fcn_unet_s5-d16_64x64_80k_drive.py'
def get_fcn_unet_s5_d16_64x64_4b_80k_drive_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet']
    if mode=='train':
        cfg = Config.fromfile(fcn_unet_s5_d16_64x64_80k_drive_config)
    else:
        raise NotImplementedError
    return cfg

def get_fcn_r18_d8_320x320_10k_voc2012_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet']
    if mode=='train':
        cfg = Config.fromfile(fcn_r18_d8_320x320_10k_voc2012_config)
    else:
        raise NotImplementedError
    return cfg

def get_fcn_r18_d8_320x320_20k_voc2012_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet']
    if mode=='train':
        cfg = Config.fromfile(root_path + 'fcn_r18-d8_512x512_b16_10k_voc2012.py')
    else:
        raise NotImplementedError
    return cfg

# voc2012 30k
def get_fcn_r18_d8_320x320_30k_voc2012_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet']
    if mode=='train':
        cfg = Config.fromfile(fcn_r18_d8_320x320_30k_voc2012_config)
    elif mode == 'usnet':
        cfg = Config.fromfile(root_path + 'fcn_r18-d8_512x512_b16_30k_voc2012_usnet.py')
    elif mode == 'fn3':
        cfg = Config.fromfile(root_path + 'fcn_r18-d8_512x512_b16_30k_voc2012_fn3.py')
    else:
        raise NotImplementedError
    return cfg

def get_fcn_r18_d8_320x320_40k_voc2012_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet']
    if mode=='train':
        cfg = Config.fromfile(fcn_r18_d8_320x320_40k_voc2012_config)
    else:
        raise NotImplementedError
    return cfg

def get_fcn_r18_d8_320x320_10k_cityscapes_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet']

    if mode=='train':
        cfg = Config.fromfile(fcn_r18_d8_320x320_10k_cityscapes_config)
    elif mode=='fine_tune':
        cfg = Config.fromfile(fcn_r18_d8_320x320_10k_cityscapes_fine_tune_config)
    elif mode=='usnet':
        cfg = Config.fromfile(fcn_r18_d8_320x320_10k_cityscapes_usnet_config)
    elif mode=='fn3':
        cfg = Config.fromfile(fcn_r18_d8_320x320_10k_cityscapes_fn3_config)
    elif mode=='cgnet':
        cfg = Config.fromfile(fcn_r18_d8_320x320_10k_cityscapes_cgnet_config)
    else:
        raise NotImplementedError

    return cfg


def get_fcn_r50_d8_512x1024_80k_cityscapes_config(input_size):
    cfg = Config.fromfile(fcn_r50_d8_512x1024_80k_cityscapes_config)
    cfg.data.legotrain.pipeline[1].img_scale = input_size[-2:]
    cfg.data.legotrain.pipeline[1].transforms[0].keep_ratio = False
    
    cfg.data.test.pipeline[1].img_scale = input_size[-2:]
    cfg.data.test.pipeline[1].transforms[0].keep_ratio = False
    return cfg

def get_fcn_r18_d8_512x1024_80k_cityscapes_config(mode='train', input_size=None):
    assert mode in ['train', 'fine_tune', 'usnet']
    if mode=='train':
        cfg = Config.fromfile(fcn_r18_d8_512x1024_80k_cityscapes_config)
    elif mode=='fine_tune':
        cfg = None
    elif mode=='usnet':
        cfg = None
    else:
        raise NotImplementedError
    if input_size is not None:
        cfg.data.train.pipeline.img_scale = input_size[-2:]
        cfg.data.legotrain.pipeline[1].img_scale = input_size[-2:]
        cfg.data.test.pipeline[1].img_scale = input_size[-2:]
        cfg.data.val.pipeline[1].img_scale = input_size[-2:]
    return cfg

def get_deeplabv3_r18_d8_512x1024_80k_cityscapes_config(input_size=None):
    
    cfg = Config.fromfile(deeplabv3_r18_d8_512x1024_80k_cityscapes_config)
    cfg.data.legotrain.pipeline[1].img_scale = input_size[-2:]
    cfg.data.legotrain.pipeline[1].transforms[0].keep_ratio = False
    
    cfg.data.test.pipeline[1].img_scale = input_size[-2:]
    cfg.data.test.pipeline[1].transforms[0].keep_ratio = False
    return cfg

def get_ccnet_r50_d8_512x1024_80k_cityscapes_config(input_size):
    cfg = Config.fromfile(ccnet_r50_d8_512x1024_80k_cityscapes_config)
    cfg.data.legotrain.pipeline[1].img_scale = input_size[-2:]
    cfg.data.legotrain.pipeline[1].transforms[0].keep_ratio = False
    
    cfg.data.test.pipeline[1].img_scale = input_size[-2:]
    cfg.data.test.pipeline[1].transforms[0].keep_ratio = False
    return cfg
    
# def get_fcn_unet_s5_d16_64x64_40k_drive_config(input_size):
#     cfg = Config.fromfile(fcn_unet_s5_d16_64x64_40k_drive_config)
#     cfg.data.legotrain.pipeline[1].img_scale = input_size[-2:]
#     cfg.data.legotrain.pipeline[1].transforms[0].keep_ratio = False
    
#     cfg.data.test.pipeline[1].img_scale = input_size[-2:]
#     cfg.data.test.pipeline[1].transforms[0].keep_ratio = False
#     return cfg

def get_gcnet_r50_d8_512x1024_40k_cityscapes_config(input_size):
    cfg = Config.fromfile(gcnet_r50_d8_512x1024_40k_cityscapes_config)
    cfg.data.legotrain.pipeline[1].img_scale = input_size[-2:]
    cfg.data.legotrain.pipeline[1].transforms[0].keep_ratio = False
    
    cfg.data.test.pipeline[1].img_scale = input_size[-2:]
    cfg.data.test.pipeline[1].transforms[0].keep_ratio = False
    return cfg

def get_emanet_r50_d8_512x1024_80k_cityscapes_config(input_size):
    cfg = Config.fromfile(emanet_r50_d8_512x1024_80k_cityscapes_config)
    cfg.data.legotrain.pipeline[1].img_scale = input_size[-2:]
    cfg.data.legotrain.pipeline[1].transforms[0].keep_ratio = False
    
    cfg.data.test.pipeline[1].img_scale = input_size[-2:]
    cfg.data.test.pipeline[1].transforms[0].keep_ratio = False
    return cfg

def get_pointrend_r50_512x1024_80k_cityscapes_config(input_size):
    cfg = Config.fromfile(pointrend_r50_512x1024_80k_cityscapes_config)
    cfg.data.legotrain.pipeline[1].img_scale = input_size[-2:]
    cfg.data.legotrain.pipeline[1].transforms[0].keep_ratio = False
    
    cfg.data.test.pipeline[1].img_scale = input_size[-2:]
    cfg.data.test.pipeline[1].transforms[0].keep_ratio = False
    return cfg

def get_dmnet_r50_d8_512x1024_80k_cityscapes_config(input_size):
    cfg = Config.fromfile(dmnet_r50_d8_512x1024_80k_cityscapes_config)
    cfg.data.legotrain.pipeline[1].img_scale = input_size[-2:]
    cfg.data.legotrain.pipeline[1].transforms[0].keep_ratio = False
    
    cfg.data.test.pipeline[1].img_scale = input_size[-2:]
    cfg.data.test.pipeline[1].transforms[0].keep_ratio = False
    return cfg

if __name__=='__main__':
    cfg = get_fcn_r18_d8_512x1024_40k_cityscapes_config()
    print(cfg.data.train)
    # print(cfg.data.legotrain.pipeline[1].img_scale)