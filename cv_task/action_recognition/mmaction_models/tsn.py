import os
import sys

import torch
sys.path.insert(0, '../../../')

from mmaction.apis import init_recognizer, inference_recognizer

from cv_task.action_recognition.mmaction_models.load_mode import LOAD_MODE
from legodnn.utils.dl.common.model import get_model_size, get_module
from cv_task.action_recognition.mmaction_models.legodnn_configs import get_tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb_config
from cv_task.action_recognition.mmaction_tools.get_input_by_size import get_input_by_size
from cv_task.datasets.action_recognition.mmaction_hmdb51 import mmaction_hmdb51_dataloader
from cv_task.action_recognition.mmaction_tools.test import test_recognizer

pretrained_root_path = '/data/gxy/pretrained_models/mmaction/'
def tsn_r50_1x1x8_50e_hmdb51_imagenet(config, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    checkpoint = os.path.join(pretrained_root_path, 'tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb_20201123-ce6c27ed.pth')
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
    model_config = get_tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb_config()
    # model = tsn_r50_1x1x8_50e_hmdb51_imagenet(model_config)
    # print(model_config.data.get('test_dataloader', {}))
    # print(dict(videos_per_gpu=1))
    # print(model)
    # input = torch.randn(1,1,3,256,256).cuda()
    # model(input)
    
    model = tsn_r50_1x1x8_50e_hmdb51_imagenet(model_config, 'mmaction_test')
    # 测试模型推理
    input = get_input_by_size(model, None)
    print("推理开始")
    with torch.no_grad():
        scores = model(return_loss=False, **input)
    print("推理结束")
    
    
    # 测试模型精度
    train_loader, test_loader = mmaction_hmdb51_dataloader(model_config, 64, 1, 1)
    test_recognizer(model, test_loader)
    pass

  