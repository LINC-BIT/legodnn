
import torch

def get_cifar100_mobilnetv2_w2_train_config_200e(mode:str='train'):
    assert mode in ['train', 'usnet', 'fn3', 'fine_tune', 'nestdnn']
    config_dict = {
        'epoch_num': 200,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 4e-5,
        'milestones': [100, 150],
        'gamma': 0.1,
        'train_batch_size': 128,
        'test_batch_size': 128,
    }
    if mode == 'train':
        pass
    elif mode == 'usnet':
        config_dict['epoch_num'] = 300
        config_dict['milestones'] = [150, 225]
    elif mode == 'fn3':
        config_dict['epoch_num'] = 500
        config_dict['milestones'] = [250, 375]
    elif mode == 'fine_tune':
        config_dict['epoch_num'] = 40
        config_dict['learning_rate'] = 0.01
        config_dict['milestones'] = [20] 
    elif mode=='nestdnn':
        config_dict['epoch_num'] = 40
        config_dict['learning_rate'] = 0.01
        config_dict['milestones'] = [20] 
        config_dict['weight_decay'] = 0.0        
    else:
        raise NotImplementedError
    return config_dict