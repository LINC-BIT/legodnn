from mmcv import Config
import sys
import os 
sys.path.insert(0, '../../../../')
root_path = 'cv_task/pose_estimation/mmpose_models/legodnn_configs/'
deeppose_res50_coco_wholebody_256x192_config = root_path + 'deeppose_res50_coco_wholebody_256x192.py'
simplebaseline_res50_coco_wholebody_256x192_config = root_path + 'simplebaseline_res50_coco_wholebody_256x192.py'
deeppose_res50_coco_256x192_config = root_path + 'deeppose_res50_coco_256x192.py'
simplebaseline_res50_coco_256x192_config = root_path + 'simplebaseline_res50_coco_256x192.py'

def get_deepose_res18_mpii_256_256_310e_128b_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet', 'nestdnn']
    basedir = os.path.abspath(os.path.dirname(__file__))
    if mode in ['train', 'cgnet']:
        cfg = Config.fromfile(os.path.join(basedir, 'deepose_res18_mpii_256x256_310e_128b.py'))
    elif mode=='usnet':
        cfg = Config.fromfile(os.path.join(basedir, 'deepose_res18_mpii_256x256_310e_128b_usnet.py'))
    elif mode=='fn3':
        cfg = Config.fromfile(os.path.join(basedir, 'deepose_res18_mpii_256x256_310e_128b_fn3.py'))
    elif mode=='fine_tune':
        cfg = Config.fromfile(os.path.join(basedir, 'deepose_res18_mpii_256x256_310e_128b_fine_tune.py'))
    elif mode=='nestdnn':
        cfg = Config.fromfile(os.path.join(basedir, 'deepose_res18_mpii_256x256_310e_128b_nestdnn.py'))
    else:
        raise NotImplementedError
    return cfg

def get_simplebaseline_res18_mpii_256_256_310e_128b_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet', 'nestdnn']
    basedir = os.path.abspath(os.path.dirname(__file__))
    if mode in ['train', 'cgnet']:
        cfg = Config.fromfile(os.path.join(basedir, 'simplebaseline_res18_mpii_256x256_310e_128b.py'))
    elif mode=='usnet':
        cfg = Config.fromfile(os.path.join(basedir, 'simplebaseline_res18_mpii_256x256_310e_128b_usnet.py'))
    elif mode=='fn3':
        cfg = Config.fromfile(os.path.join(basedir, 'simplebaseline_res18_mpii_256x256_310e_128b_fn3.py'))
    elif mode=='fine_tune':
        cfg = Config.fromfile(os.path.join(basedir, 'simplebaseline_res18_mpii_256x256_310e_128b_fine_tune.py'))
    elif mode=='nestdnn':
        cfg = Config.fromfile(os.path.join(basedir, 'simplebaseline_res18_mpii_256x256_310e_128b_nestdnn.py'))
    else:
        raise NotImplementedError
    return cfg

def get_deepose_res18_mpii_256_256_210e_128b_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet']
    if mode=='train':
        cfg = Config.fromfile(root_path + '/deepose_res18_mpii_256x256_210e_128b.py')
    else:
        raise NotImplementedError
    return cfg

def get_deepose_res18_mpii_256_256_315e_128b_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet']
    if mode=='train':
        cfg = Config.fromfile(root_path + '/deepose_res18_mpii_256x256_315e_128b.py')
    else:
        raise NotImplementedError
    return cfg

def get_deepose_res18_mpii_256_256_420e_128b_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet']
    if mode=='train':
        cfg = Config.fromfile(root_path + '/deepose_res18_mpii_256x256_420e_128b.py')
    else:
        raise NotImplementedError
    return cfg

def get_deepose_res18_coco_256_192_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet']
    if mode=='train':
        cfg = Config.fromfile(root_path + '/base_deeppose_res18_coco_256x192.py')
    else:
        raise NotImplementedError
    return cfg

def get_deepose_res18_mpii_256_256_config(mode='train'):
    assert mode in ['train', 'fine_tune', 'usnet', 'fn3', 'cgnet']
    if mode=='train':
        cfg = Config.fromfile(root_path + '/base_deepose_res18_mpii_256x256.py')
    else:
        raise NotImplementedError
    return cfg

def get_deeppose_res50_coco_wholebody_256x192_config():
    cfg = Config.fromfile(deeppose_res50_coco_wholebody_256x192_config)
    return cfg

def get_deeppose_res50_coco_256x192_config():
    cfg = Config.fromfile(deeppose_res50_coco_256x192_config)
    return cfg

def get_simplebaseline_res50_coco_wholebody_256x192_config():
    cfg = Config.fromfile(simplebaseline_res50_coco_wholebody_256x192_config)
    return cfg

def get_simplebaseline_res50_coco_256x192_config():
    cfg = Config.fromfile(simplebaseline_res50_coco_256x192_config)
    return cfg


if __name__=='__main__':
    # cfg = get_fcn_unet_s5_d16_64x64_40k_drive_config(input_size=(224, 224))
    # # print(a)
    # print(cfg.data.legotrain.pipeline[1].img_scale)
    pass