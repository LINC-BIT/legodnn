from copy import copy
import torch
import tqdm
import sys
import copy
sys.path.insert(0, '../../')

from legodnn.utils.dl.common.model import get_model_flops_and_params, get_model_latency, get_model_size, get_model_flops_and_params_by_dummy_input, get_model_device
from legodnn.abstract_model_manager import AbstractModelManager
from mmdet.apis import single_gpu_test
from mmcv.parallel import MMDataParallel
from mmcv.parallel.scatter_gather import scatter

from cv_task.pose_estimation.mmpose_tools.test import test_posenet
from cv_task.pose_estimation.mmpose_tools.get_input_by_size import get_input_by_size
from cv_task.pose_estimation.mmpose_models.deeppose import deeppose_res50_coco_256x192
from cv_task.pose_estimation.mmpose_models.legodnn_configs import get_deeppose_res50_coco_256x192_config
from cv_task.datasets.pose_estimation.mmpose_coco import mmpose_coco_dataloader



class CommonPoseEstimationModelManager(AbstractModelManager):
    def forward_to_gen_mid_data(self, model, batch_data, device):
        with torch.no_grad():
            model.val_step(batch_data, None)

    def dummy_forward_to_gen_mid_data(self, model, model_input_size, device):
        # batch_data = (torch.rand(model_input_size).to(device), None)
        # self.forward_to_gen_mid_data(model, get_input_by_size(model, model_input_size), device)
        batch_data = get_input_by_size(model)
        with torch.no_grad():
            # model(return_loss=False, rescale=False, **get_input_by_size(model, model_input_size))
            model(
                img=batch_data['img'],
                img_metas=batch_data['img_metas'],
                return_loss=False,
                return_heatmap=False)

    def get_model_acc(self, model, test_loader, device='cuda'):
        acc = test_posenet(model, test_loader)
        return float(acc)

    def get_model_size(self, model):
        return get_model_size(model)

    def get_model_flops_and_param(self, model, model_input_size):
        model = copy.deepcopy(model)
        model.forward = model.forward_dummy
        return get_model_flops_and_params(model, model_input_size)
    
    # def get_model_flops_and_param(self, model, model_input_size):
    #     device = get_model_device(model)
    #     dummy_input = (torch.ones(model_input_size).to(device), )
    #     return get_model_flops_and_params_by_dummy_input(model, dummy_input)
    
    def get_model_latency(self, model, sample_num, model_input_size, device):
        model = copy.deepcopy(model)
        model.forward = model.forward_dummy
        return get_model_latency(model, model_input_size, sample_num, device, sample_num // 2)

if __name__=='__main__':
    model_config = get_deeppose_res50_coco_256x192_config()
    teacher_segmentor = deeppose_res50_coco_256x192(model_config, mode='mmpose_test', device='cuda')
    train_loader, test_loader = mmpose_coco_dataloader(model_config)
    model_manger = CommonPoseEstimationModelManager()
    model_manger.get_model_acc(teacher_segmentor, test_loader)