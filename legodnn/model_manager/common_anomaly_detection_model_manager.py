import torch
import tqdm

from ..utils.dl.common.model import get_model_flops_and_params, get_model_latency, get_model_size
from ..abstract_model_manager import AbstractModelManager
from cv_task.anomaly_detection.models_tools.test import test_ganomaly_model, test_gpnd_model, test_ornet_model

MODEL_NAME = ['ganomaly', 'gpnd', 'ornet']

class CommonAnomalyDetectionModelManager(AbstractModelManager):
    def __init__(self, model_name, train_loader=None, all_video_frames_label=None):
        super().__init__()
        assert model_name in MODEL_NAME
        self.model_name = model_name
        self.train_loader = train_loader
        self.all_video_frames_label = all_video_frames_label
        
    def forward_to_gen_mid_data(self, model, batch_data, device):
        model = model.to(device)
        data = batch_data[0].to(device)
        model.eval()
        with torch.no_grad():
            model(data)
            
    def dummy_forward_to_gen_mid_data(self, model, model_input_size, device):
        batch_data = (torch.rand(model_input_size).to(device), None)
        self.forward_to_gen_mid_data(model, batch_data, device)
    
    def get_model_acc(self, model, test_loader, device='cuda'):
        if self.model_name == 'ganomaly':
            acc = test_ganomaly_model(model, test_loader, dataset_name='coil100', device=device)
        elif self.model_name == 'gpnd':
            acc = test_gpnd_model(model, self.train_loader, test_loader, dataset_name='caltech256', device=device)
        elif self.model_name == 'ornet':
            acc = test_ornet_model(model, test_loader, all_video_frames_label=self.all_video_frames_label)
        return acc
    
    def get_model_size(self, model):
        return get_model_size(model)

    def get_model_flops_and_param(self, model, model_input_size):
        return get_model_flops_and_params(model, model_input_size)
    
    def get_model_latency(self, model, sample_num, model_input_size, device):
        return get_model_latency(model, model_input_size, sample_num, device, sample_num // 2)
