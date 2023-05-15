import os
import sys

import torch
sys.path.insert(0, '../../../')

from mmaction.apis import init_recognizer, inference_recognizer

from cv_task.action_recognition.mmaction_models.load_mode import LOAD_MODE
from legodnn.utils.dl.common.model import get_model_size, get_module
from cv_task.action_recognition.mmaction_models.legodnn_configs import get_slowfast_r50_16x8x1_22e_sthv1_rgb_config
from cv_task.action_recognition.mmaction_tools.get_input_by_size import get_input_by_size
# from cv_task.datasets.action_recognition.mmaction_hmdb51 import mmaction_hmdb51_dataloader
from cv_task.action_recognition.mmaction_tools.test import test_recognizer

pretrained_root_path = '/data/gxy/pretrained_models/mmaction/'

def slowfast_r50_16x8x1_22e_sthv1(config, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    checkpoint = os.path.join(pretrained_root_path, 'slowfast_r50_16x8x1_22e_sthv1_rgb_20210630-53355c16.pth')
    if mode=='lego_jit':
        recognizer = init_recognizer(config, checkpoint, device=device)
        recognizer.forward = recognizer.forward_dummy
        
    elif mode=='mmaction_test':
        recognizer = init_recognizer(config, checkpoint, device=device)
        recognizer = recognizer
    else:
        raise NotImplementedError
    recognizer.eval()
    return recognizer

if __name__=='__main__':
    model_config = get_slowfast_r50_16x8x1_22e_sthv1_rgb_config()
    model = slowfast_r50_16x8x1_22e_sthv1(model_config)
    print('model size {:.3f}MB'.format(get_model_size(model) / 1024**2))
    print('backbone size {:.3f}MB'.format(get_model_size(get_module(model, 'backbone')) / 1024**2))
    print('cls_head size {:.3f}MB'.format(get_model_size(get_module(model, 'cls_head')) / 1024**2))
    print(model)
    # exit(0)
    # input = torch.randn(1,64,3,256,256).cuda()
    # model(input)
    # exit(0)
    model = slowfast_r50_16x8x1_22e_sthv1(model_config, 'mmaction_test')
    # 测试模型推理
    input = get_input_by_size(model, None)
    print("推理开始")
    with torch.no_grad():
        scores = model(return_loss=False, **input)
    print("推理结束")
    
    
    # # 测试模型精度
    # train_loader, test_loader = mmaction_hmdb51_dataloader(model_config, 64, 1, 1)
    # test_recognizer(model, test_loader)
    # pass

  