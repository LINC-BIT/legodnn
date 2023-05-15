from typing import Dict
import torch
from torch._C import device
from torch.functional import block_diag
import torch.nn as nn
import copy

from torch.nn.modules import module

from legodnn.block_detection.block_detection import BlockDetection
from legodnn.block_detection.model_topology_extraction import LegoDNNGraph, LegoDNNNode, topology_extraction
from legodnn.utils.dl.common.pruning import l1_prune_model_by_dummy_input
from legodnn.abstract_block_manager import AbstractBlockManager
from legodnn.utils.dl.common.model import get_ith_layer, get_model_flops_and_params, get_model_latency, get_module, save_model, ModelSaveMethod, \
    LayerActivation, get_model_size, TimeProfiler, LayerActivationWrapper, TimeProfilerWrapper, set_module, get_model_flops_and_params_by_dummy_input, get_model_latency_by_dummy_input, get_all_specific_type_layers_name
from legodnn.utils.dl.common.model import ReuseLayerActivation, get_model_device
from legodnn.utils.common.file import ensure_dir
from legodnn.block_detection.block_extraction_11_28 import LegoDNNBlock
# from gxy_test.block_extraction import LegoDNNBlock
# from gxy_test.detection_manager import ObjectDeteDetectionManager


class AutoBlockManager(AbstractBlockManager):
    def __init__(self, block_sparsity, detection_manager, model_manager):
        self.graph = detection_manager.graph
        self.blocks = detection_manager.get_blocks()
        self.block_num = len(self.blocks)

        # 自动生成块id
        blocks_id = []
        for i in range(0, self.block_num):
            blocks_id.append('block-{}'.format(i))
        # blocks_id = detection_manager.get_blocks_id()
        self.detection_manager = detection_manager
        # 生成块数个稀疏度列表
        blocks_sparsity = [block_sparsity for _ in range(self.block_num)]
        super(AutoBlockManager, self).__init__(blocks_id, blocks_sparsity, model_manager)

    def get_block_from_model(self, model, block_id):
        # 根据block_id抽取出块
        num_in_block = self.detection_manager.get_num_in_block(block_id)
        block_info = self.detection_manager.get_block_io_info(block_id)  # 包括 占位符 开始节点 结束节点
        legodnn_block = LegoDNNBlock(model, num_in_block, self.graph, block_info)
        return legodnn_block

    # def set_block_to_model(self, model, block_id, block):
    #     print(model)
    #     print(block)
    #     for name, module in block.named_modules():
    #         if len(list(module.children())) > 0:
    #             continue

    #         # module_name = self.detection_manager.get_module_name(name)
    #         # print('name {}'.format(name))
    #         # print('module name {}'.format(module_name))
    #         # assert name==module_name
    #         block_module = copy.deepcopy(module)
    #         model_module = get_module(model, name)
    #         assert model_module is not None
            
    #         set_module(model, name, block_module)

    def set_block_to_model(self, model, block_id, block):
        # print(model)
        # print(block)
        # flag = False
        for name, module in block.named_modules():
            if len(list(module.children())) > 0:
                continue
            block_module = copy.deepcopy(module)
            # model_module = get_module(model, name)
            # flag =  model_module is None
            set_module(model, name, block_module) # 这里可能会出问题，teachermodel和原始模型对不上的问题

    def empty_block_in_model(self, model, block_id):
        # 根据block_id抽取出块
        num_in_block = self.detection_manager.get_num_in_block(block_id)
        block_info = self.detection_manager.get_block_io_info(block_id)  # 包括 占位符 开始节点 结束节点
        empty_block = LegoDNNBlock(model, num_in_block, self.graph, block_info, is_empty=True)
        self.set_block_to_model(model, block_id, empty_block)


    def get_default_blocks_id(self):
        return self._blocks_id

    def get_pruned_block(self, model, block_id, block_sparsity, model_input_size, device):
        model = model.to(device)
        num_in_block = self.detection_manager.get_num_in_block(block_id)
        # 从模型中提取出LegoDNNBlock类型的块
        raw_block = self.get_block_from_model(model, block_id)
        print("第{}个原始块".format(block_id))
        print(raw_block)
        # 稀疏度为0则就是原始块
        if block_sparsity == 0.0:
            return copy.deepcopy(raw_block)
 
        # 找出块中的卷积操作 记录下其名字
        pruned_conv_layers = []
        no_compressed_layers = self.detection_manager.get_no_compressed_layers(block_id)
        for num in num_in_block:
            if self.graph.order_to_node.get(num).get_op_type() in ['Conv2d', 'ConvTranspose2d']: # 目前压缩层去掉线性层，因为nni的l1剪枝目前不支持，后续有时间可以自己实现
                module_name = self.detection_manager.get_module_name(num)
                # 不在非可压缩层中的卷积层才加入到列表中
                if module_name not in no_compressed_layers:
                    pruned_conv_layers += [module_name]
        
        # 处理低秩分解的情况
        tmp_pruned_conv_layers = []
        for pruned_conv_name in pruned_conv_layers:
            block_module = get_module(raw_block, pruned_conv_name)
            if isinstance(block_module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                tmp_pruned_conv_layers.append(pruned_conv_name)
            else:
                name_list = get_all_specific_type_layers_name(block_module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))
                assert len(name_list)>=1
                tmp_pruned_conv_layers = tmp_pruned_conv_layers + [pruned_conv_name+'.'+name for name in name_list]
                
        pruned_conv_layers = tmp_pruned_conv_layers
        
        # print(pruned_conv_layers)
        
        # 找块的输入形状 也就是勾出的输入数据形状 要根据self.block_info中信息找
        layer_activation_list = []
        start_module_name_list = self.detection_manager.get_blocks_start_node_name_hook(block_id)
        for start_module_name in start_module_name_list:
            layer_activation_list.append(ReuseLayerActivation(get_module(model, start_module_name), device))
        
        self._model_manager.dummy_forward_to_gen_mid_data(model, model_input_size, device)
        need_hook_input_list = []
        start_node_hook_input_or_ouput_list =  self.detection_manager.get_blocks_start_node_hook_input_or_ouput(block_id)
        for start_node_hook_input_or_ouput in start_node_hook_input_or_ouput_list:
            need_hook_input_list.append(start_node_hook_input_or_ouput == 0)
        start_hook_index_list = self.detection_manager.get_blocks_start_node_hook_index(block_id)
        input_size = ()
        for i, layer_activation in enumerate(layer_activation_list):
            input_size = input_size + (layer_activation.input_list[start_hook_index_list[i]].size() if need_hook_input_list[i] else layer_activation.output_list[start_hook_index_list[i]].size(),)
            layer_activation.remove()

        if len(start_module_name_list) == 1:
            input_size = input_size[0]
        
        # 已知[原始块 需要剪枝的卷积层 被剪到的稀疏度 输入形状 设备]        对块进行对应稀疏度的剪枝 返回派生块
        print('\t--> pruned conv layers: {} input size: {}'.format(pruned_conv_layers, input_size))

        # 准备需要剪枝的模型：一个输入输出通道为块输入特征图个数，卷积核为1的conv2d + 当前LegoDNN块
        prepare_model = nn.Sequential()
        if len(layer_activation_list) == 1:
            input_channels = input_size[1]
            prepare_model.add_module('pruning_init_conv', torch.nn.Conv2d(input_channels, input_channels, kernel_size=1))
            prepare_model.add_module('legodnn_block', raw_block)
            prepare_model.to(device)
            input_data = torch.rand(input_size).to(device)
        elif len(layer_activation_list) > 1:
            # input_channels = input_size[1]
            # prepare_model.add_module('pruning_init_conv', torch.nn.Conv2d(input_channels, input_channels, kernel_size=1))
            prepare_model.add_module('legodnn_block', raw_block)
            prepare_model.to(device)
            input_data = ()
            for tensor_size in input_size:
                input_data = input_data + (torch.rand(tensor_size).to(device), )
            input_data = (input_data, )
        # print(prepare_model)

        # 剪枝模型，首先处理剪枝的层，都需要加上'legodnn_block'
        pruned_conv_layers = ['legodnn_block.'+module_name for module_name in pruned_conv_layers]

        # # 已知[原始块 需要剪枝的卷积层 被剪到的稀疏度 输入形状 设备]        对块进行对应稀疏度的剪枝 返回派生块
        # print('\t--> pruned conv layers: {} input size: {}'.format(pruned_conv_layers, input_size))
        # print(prepare_model)
        pruned_model = l1_prune_model_by_dummy_input(prepare_model, pruned_conv_layers, block_sparsity, input_data, device)
        # 提取出被剪枝的块
        pruned_block = copy.deepcopy(get_module(pruned_model, 'legodnn_block'))
        # print(pruned_block)
        
        return pruned_block

    # def get_module_name_from_
    def get_io_activation_of_all_blocks(self, model, device):
        layer_activation_dict = {}
        blocks_id = self.get_blocks_id()
        for block_id in blocks_id:
            # 所有块的输入
            for module_name in self.detection_manager.get_blocks_start_node_name_hook(block_id):
                module_name = self.detection_manager.get_module_name(module_name)
                if module_name not in layer_activation_dict.keys():
                    layer_activation_dict.update({module_name: ReuseLayerActivation(get_module(model, module_name), device)})

            # 所有块的输出
            for module_name in self.detection_manager.get_blocks_end_node_name_hook(block_id):
                module_name = self.detection_manager.get_module_name(module_name)
                if module_name not in layer_activation_dict.keys():
                    layer_activation_dict.update({module_name: ReuseLayerActivation(get_module(model, module_name), device)})
        return layer_activation_dict

    def clear_io_activations(self, io_activation: Dict):
        for layer_activation in io_activation.values():
            layer_activation.clear()

    def remove_io_activations(self, io_activation: Dict):
        for layer_activation in io_activation.values():
            layer_activation.remove()

    def get_time_profilers_of_all_blocks(self, model, device):
        res = []
        # 对于每个块
        for block_id in self.get_blocks_id():
            # num_in_block = self.blocks[int(block_id.split('-')[-1])]
            num_in_block = self.detection_manager.get_num_in_block(block_id)
            time_profiler_list = []
            for num in num_in_block:
                node = self.graph.order_to_node.get(num)
                # 只有module类型才能用get_module
                if node.get_type() == 'module':
                    # module_name = node.get_name()
                    # module_name = self.block_info._find_module_node_in_model_name(module_name)
                    # module_name = self.block_info._find_module_node_in_model_name(num)
                    module_name = self.detection_manager.get_module_name(num)
                    time_profiler_list.append(TimeProfiler(get_module(model, module_name), device))
            res += [TimeProfilerWrapper(time_profiler_list)]
        return res

    def get_block_file_name(self, block_id, block_sparsity):
        return '{}-{}.pt'.format(block_id, str(block_sparsity).split('.')[-1])


    def save_block_to_file(self, block, block_file_path):
        ensure_dir(block_file_path)
        save_model(block, block_file_path, ModelSaveMethod.FULL)


    def get_block_from_file(self, block_file_path, device):
        return torch.load(block_file_path, map_location=device)

    # def should_continue_train_block(self, last_loss, cur_loss):
    #     if cur_loss < 1e-8:
    #         return False
    #     return (last_loss - cur_loss) / last_loss > 1e-3
    
    def should_continue_train_block(self, last_loss, cur_loss):
        # if cur_loss < 1e-8:
        #     return False
        return True


    def get_block_size(self, block):
        return get_model_size(block)
    

    # def get_block_flops_and_params(self, block, input_size):
        # return get_model_flops_and_params(block, input_size)
        
    def get_block_flops_and_params(self, block, dummy_input):
        return get_model_flops_and_params_by_dummy_input(block, dummy_input)
     

    # def get_block_latency(self, block, sample_num, input_size, device):
    #     return get_model_latency(block, input_size, sample_num, device, sample_num)

    def get_block_latency(self, block, sample_num, dummy_input, device):
        return get_model_latency_by_dummy_input(block, dummy_input, sample_num, device, sample_num)