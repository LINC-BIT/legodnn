from mmcv import Config
import sys
import os
sys.path.insert(0, '../../../../')
root_path = 'cv_task/action_recognition/mmaction_models/legodnn_configs/'
    

tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb_config = root_path + 'tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb.py'
trn_r50_1x1x8_50e_sthv2_rgb_config = root_path + 'trn_r50_1x1x8_50e_sthv2_rgb.py'
tsm_k400_pretrained_r50_1x1x8_25e_hmdb51_rgb_config = root_path + 'tsm_k400_pretrained_r50_1x1x8_25e_hmdb51_rgb.py'
slowfast_r50_16x8x1_22e_sthv1_rgb_config = root_path + 'slowfast_r50_16x8x1_22e_sthv1_rgb.py'

def get_tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb_config():
    cfg = Config.fromfile(tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb_config)
    return cfg


def get_tsn_r18_hmdb51_1x1x8_imagenet_rgb_50e_512b_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet']
    basedir = os.path.abspath(os.path.dirname(__file__))
    if mode in ['train', 'cgnet']:
        cfg = Config.fromfile(os.path.join(basedir, 'deepose_res18_mpii_256x256_310e_128b.py'))
    elif mode=='usnet':
        cfg = Config.fromfile(os.path.join(basedir, 'deepose_res18_mpii_256x256_310e_128b_usnet.py'))
    elif mode=='fn3':
        cfg = Config.fromfile(os.path.join(basedir, 'deepose_res18_mpii_256x256_310e_128b_fn3.py'))
    elif mode=='fine_tune':
        cfg = Config.fromfile(os.path.join(basedir, 'deepose_res18_mpii_256x256_310e_128b_fine_tune.py'))
    else:
        raise NotImplementedError
    return cfg

def get_tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet', 'nestdnn']
    basedir = os.path.abspath(os.path.dirname(__file__))
    if mode in ['train', 'cgnet']:
        cfg = Config.fromfile(os.path.join(basedir, 'tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b.py'))
    elif mode=='usnet':
        cfg = Config.fromfile(os.path.join(basedir, 'tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_usnet.py'))
    elif mode=='fn3':
        cfg = Config.fromfile(os.path.join(basedir, 'tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_fn3.py'))
    elif mode=='fine_tune':
        cfg = Config.fromfile(os.path.join(basedir, 'tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_fine_tune.py'))
    elif mode=='nestdnn':
        cfg = Config.fromfile(os.path.join(basedir, 'tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_nestdnn.py'))
    else:
        raise NotImplementedError
    return cfg

def get_trn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet', 'nestdnn']
    basedir = os.path.abspath(os.path.dirname(__file__))
    if mode in ['train', 'cgnet']:
        cfg = Config.fromfile(os.path.join(basedir, 'trn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b.py'))
    elif mode=='usnet':
        cfg = Config.fromfile(os.path.join(basedir, 'trn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_usnet.py'))
    elif mode=='fn3':
        cfg = Config.fromfile(os.path.join(basedir, 'tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_fn3.py'))
    elif mode=='fine_tune':
        cfg = Config.fromfile(os.path.join(basedir, 'tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_fine_tune.py'))
    elif mode=='nestdnn':
        cfg = Config.fromfile(os.path.join(basedir, 'tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_nestdnn.py'))
    else:
        raise NotImplementedError
    return cfg

def get_trn_r18_hmdb51_1x1x8_imagenet_rgb_50e_64b_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet', 'nestdnn']
    basedir = os.path.abspath(os.path.dirname(__file__))
    if mode in ['train', 'cgnet']:
        cfg = Config.fromfile(os.path.join(basedir, 'trn_r18_hmdb51_1x1x8_imagenet_rgb_50e_64b.py'))
    # elif mode=='usnet':
    #     cfg = Config.fromfile(os.path.join(basedir, 'tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_usnet.py'))
    # elif mode=='fn3':
    #     cfg = Config.fromfile(os.path.join(basedir, 'tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_fn3.py'))
    # elif mode=='fine_tune':
    #     cfg = Config.fromfile(os.path.join(basedir, 'tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_fine_tune.py'))
    # elif mode=='nestdnn':
    #     cfg = Config.fromfile(os.path.join(basedir, 'tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_nestdnn.py'))
    else:
        raise NotImplementedError
    return cfg

def get_trn_r18_hmdb51_1x1x8_imagenet_rgb_60e_64b_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet', 'nestdnn']
    basedir = os.path.abspath(os.path.dirname(__file__))
    if mode in ['train', 'cgnet']:
        cfg = Config.fromfile(os.path.join(basedir, 'trn_r18_hmdb51_1x1x8_imagenet_rgb_60e_64b.py'))
    elif mode=='usnet':
        cfg = Config.fromfile(os.path.join(basedir, 'trn_r18_hmdb51_1x1x8_imagenet_rgb_60e_64b_usnet.py'))
    elif mode=='fn3':
        cfg = Config.fromfile(os.path.join(basedir, 'trn_r18_hmdb51_1x1x8_imagenet_rgb_60e_64b_fn3.py'))
    elif mode=='fine_tune':
        cfg = Config.fromfile(os.path.join(basedir, 'trn_r18_hmdb51_1x1x8_imagenet_rgb_60e_64b_fine_tune.py'))
    elif mode=='nestdnn':
        cfg = Config.fromfile(os.path.join(basedir, 'trn_r18_hmdb51_1x1x8_imagenet_rgb_60e_64b_nestdnn.py'))
    else:
        raise NotImplementedError
    return cfg

def get_trn_r18_hmdb51_1x1x8_imagenet_rgb_100e_64b_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet', 'nestdnn']
    basedir = os.path.abspath(os.path.dirname(__file__))
    if mode in ['train', 'cgnet']:
        cfg = Config.fromfile(os.path.join(basedir, 'trn_r18_hmdb51_1x1x8_imagenet_rgb_100e_64b.py'))
    # elif mode=='usnet':
    #     cfg = Config.fromfile(os.path.join(basedir, 'tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_usnet.py'))
    # elif mode=='fn3':
    #     cfg = Config.fromfile(os.path.join(basedir, 'tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_fn3.py'))
    # elif mode=='fine_tune':
    #     cfg = Config.fromfile(os.path.join(basedir, 'tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_fine_tune.py'))
    # elif mode=='nestdnn':
    #     cfg = Config.fromfile(os.path.join(basedir, 'tsn_r18_hmdb51_1x1x8_imagenet_rgb_35e_64b_nestdnn.py'))
    else:
        raise NotImplementedError
    return cfg

def get_trn_r50_1x1x8_50e_sthv2_rgb_config():
    cfg = Config.fromfile(trn_r50_1x1x8_50e_sthv2_rgb_config)
    return cfg

def get_tsm_k400_pretrained_r50_1x1x8_25e_hmdb51_rgb_config():
    cfg = Config.fromfile(tsm_k400_pretrained_r50_1x1x8_25e_hmdb51_rgb_config)
    return cfg

def get_slowfast_r50_16x8x1_22e_sthv1_rgb_config():
    cfg = Config.fromfile(slowfast_r50_16x8x1_22e_sthv1_rgb_config)
    return cfg


if __name__=='__main__':
    cfg = get_tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb_config(input_size=(224, 224))
    # print(a)
    # print(cfg.data.legotrain.pipeline[1].img_scale)
