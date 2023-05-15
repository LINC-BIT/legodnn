import torch
import torch.nn
import torch.nn as nn
import copy

from ..utils.dl.common.pruning import l1_prune_model
from ..abstract_block_manager import AbstractBlockManager
from ..utils.dl.common.model import get_ith_layer, get_model_flops_and_params, get_model_latency, get_module, save_model, ModelSaveMethod, \
    LayerActivation, get_model_size, TimeProfiler, LayerActivationWrapper, TimeProfilerWrapper, set_module
from ..utils.common.file import ensure_dir


class CommonBlockManager(AbstractBlockManager):
    def get_block_from_model(self, model, block_id):
        return nn.Sequential(*[get_module(model, i) for i in block_id.split('|')])

    def set_block_to_model(self, model, block_id, block: torch.nn.Sequential):
        [set_module(model, i, p) for i, p in zip(block_id.split('|'), block.children())] 

    def empty_block_in_model(self, model, block_id):
        empty_block = torch.nn.Sequential(*[torch.nn.Sequential() for _ in range(len(block_id.split('|')))])
        self.set_block_to_model(model, block_id, empty_block)

    def get_pruned_block(self, model, block_id, block_sparsity, model_input_size, device):
        model = model.to(device)
        # 从模型中提取出块
        raw_block = self.get_block_from_model(model, block_id)
        
        if block_sparsity == 0.0:
            return copy.deepcopy(raw_block)

        pruned_conv_layers = []
        for name, module in raw_block.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                pruned_conv_layers += [name]
        pruned_conv_layers = pruned_conv_layers[0:-1]
        
        # 模型输入
        layer_activation = LayerActivation(get_ith_layer(raw_block, 0), device)
        self._model_manager.dummy_forward_to_gen_mid_data(model, model_input_size, device)
        input_size = layer_activation.input.size()
        layer_activation.remove()

        return l1_prune_model(raw_block, pruned_conv_layers, block_sparsity, input_size, device)

    def get_io_activation_of_all_blocks(self, model, device):
        res = []
        for block_id in self.get_blocks_id():
            res += [LayerActivationWrapper([
                LayerActivation(get_module(model, i), device) for i in block_id.split('|')
            ])]

        return res

    def get_time_profilers_of_all_blocks(self, model, device):
        res = []
        for block_id in self.get_blocks_id():
            res += [TimeProfilerWrapper([
                TimeProfiler(get_module(model, i), device) for i in block_id.split('|')
            ])]

        return res

    def get_block_file_name(self, block_id, block_sparsity):
        return '{}-{}.pt'.format(block_id.replace('.', '_'), str(block_sparsity).split('.')[-1])

    def save_block_to_file(self, block, block_file_path):
        ensure_dir(block_file_path)
        save_model(block, block_file_path, ModelSaveMethod.FULL)

    def get_block_from_file(self, block_file_path, device):
        return torch.load(block_file_path, map_location=device)

    def should_continue_train_block(self, last_loss, cur_loss):
        if cur_loss < 1e-8:
            return False
        return (last_loss - cur_loss) / last_loss > 1e-3

    def get_block_size(self, block):
        return get_model_size(block)
    
    def get_block_flops_and_params(self, block, input_size):
        return get_model_flops_and_params(block, input_size)
     
    def get_block_latency(self, block, sample_num, input_size, device):
        return get_model_latency(block, input_size, sample_num, device, sample_num // 3)