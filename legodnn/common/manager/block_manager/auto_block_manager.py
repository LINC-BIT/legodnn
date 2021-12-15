from typing import Dict
import torch
from torch._C import device
import torch.nn as nn
import copy,os
from legodnn.common.utils.common.log import logger

# from legodnn.block_detection.block_detection import BlockDetection
# from legodnn.block_detection.model_topology_extraction import LegoDNNGraph, LegoDNNNode, topology_extraction
from legodnn.common.utils.dl.common.pruning import l1_prune_model_by_dummy_input
from legodnn.common.manager.block_manager.abstract_block_manager import AbstractBlockManager
from legodnn.common.utils.dl.common.model import get_module, save_model, ModelSaveMethod, \
    get_model_size, TimeProfiler, TimeProfilerWrapper, set_module, get_model_flops_and_params_by_dummy_input, get_model_latency_by_dummy_input
from legodnn.common.utils.dl.common.model import ReuseLayerActivation
from legodnn.common.utils.common.file import ensure_dir
from legodnn.common.detection.block_extraction_11_28 import LegoDNNBlock
from legodnn.common.detection.model_topology_extraction import topology_extraction
from legodnn.common.detection.common_detection_manager_1204_new import CommonDetectionManager
class AutoBlockManager(AbstractBlockManager):
    def __init__(self, block_sparsity, teacher_model, model_manager,model_input_size,max_ratio,device=device,):
        self.blocks = self.get_blocks(teacher_model,model_input_size,device,max_ratio)
        self.block_num = len(self.blocks)
        self.model_input_size = model_input_size
        self.device = device
        self.teacher_model = teacher_model
        # 自动生成块id
        blocks_id = []
        for i in range(0, self.block_num):
            blocks_id.append('block-{}'.format(i))

        print('\033[1;36m-------------------------------->    BUILD LEGODNN GRAPH\033[0m')


        # 生成块数个稀疏度列表
        blocks_sparsity = [block_sparsity for _ in range(self.block_num)]
        super(AutoBlockManager, self).__init__(blocks_id, blocks_sparsity, model_manager)

    # block_extractor
    def get_blocks(self,teacher_model,model_input_size,device,max_ratio):
        model_graph = topology_extraction(teacher_model, model_input_size, device=device)
        model_graph.print_ordered_node()
        # exit(0)
        print('\033[1;36m-------------------------------->    START BLOCK DETECTION\033[0m')

        # block_extractor
        detection_manager = CommonDetectionManager(model_graph, max_ratio=max_ratio)  # resnet18
        detection_manager.detection_all_blocks()
        detection_manager.print_all_blocks()
        self.graph = detection_manager.graph
        self.detection_manager = detection_manager
        return detection_manager.get_blocks()

    def get_block_from_model(self, model, block_id):
        # 根据block_id抽取出块
        num_in_block = self.detection_manager.get_num_in_block(block_id)
        block_info = self.detection_manager.get_block_io_info(block_id)  # 包括 占位符 开始节点 结束节点
        legodnn_block = LegoDNNBlock(model, num_in_block, self.graph, block_info)
        return legodnn_block

    # Block Switcher
    def set_block_to_model(self, model, block_id, block):
        for name, module in block.named_modules():
            if len(list(module.children())) > 0:
                continue

            module_name = self.detection_manager.get_module_name(name)
            assert name==module_name

            block_module = copy.deepcopy(module)
            set_module(model, name, block_module)

    def empty_block_in_model(self, model, block_id):
        # 根据block_id抽取出块
        num_in_block = self.detection_manager.get_num_in_block(block_id)
        block_info = self.detection_manager.get_block_io_info(block_id)  # 包括 占位符 开始节点 结束节点
        empty_block = LegoDNNBlock(model, num_in_block, self.graph, block_info, is_empty=True)
        self.set_block_to_model(model, block_id, empty_block)


    def get_default_blocks_id(self):
        return self._blocks_id

    # Descendant Block Generator
    def get_pruned_block(self, model, block_id, block_sparsity, model_input_size, device):
        model = model.to(device)
        num_in_block = self.detection_manager.get_num_in_block(block_id)
        # 从模型中提取出LegoDNNBlock类型的块
        raw_block = self.get_block_from_model(model, block_id)

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
        print(pruned_block)
        return pruned_block

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

    def should_continue_train_block(self, last_loss, cur_loss):
        if cur_loss < 1e-8:
            return False
        return (last_loss - cur_loss) / last_loss > 1e-3

    def get_block_size(self, block):
        return get_model_size(block)
        
    def get_block_flops_and_params(self, block, dummy_input):
        return get_model_flops_and_params_by_dummy_input(block, dummy_input)

    def get_block_latency(self, block, sample_num, dummy_input, device):
        return get_model_latency_by_dummy_input(block, dummy_input, sample_num, device, sample_num)

    # block_extractor
    def _save_compressed_block(self, model, block_id, block_sparsity,block_save_dir):
        compressed_block = self.get_pruned_block(model, block_id, block_sparsity,
                                                                self.model_input_size, self.device)
        # exit(0)
        pruned_block_file_path = os.path.join(block_save_dir,
                                              self.get_block_file_name(block_id, block_sparsity))
        self.save_block_to_file(compressed_block, pruned_block_file_path)
        logger.info('save pruned block {} (sparsity {}) in {}'.format(block_id, block_sparsity, pruned_block_file_path))
        logger.debug(compressed_block)

    def _compress_single_block(self, block_id, block_sparsity,block_save_dir):
        # 深拷贝模型供压缩用
        model = copy.deepcopy(self.teacher_model)
        model = model.to(self.device)
        self._save_compressed_block(model, block_id, block_sparsity,block_save_dir)

    def _save_model_frame(self,block_save_dir):
        # model frame
        empty_model = copy.deepcopy(self.teacher_model)
        # print(empty_model)
        for block_id in self.get_blocks_id():
            self.empty_block_in_model(empty_model, block_id)
        model_frame_path = os.path.join(block_save_dir, 'model_frame.pt')
        ensure_dir(model_frame_path)
        # print(empty_model)
        # exit(0)
        save_model(empty_model, model_frame_path, ModelSaveMethod.FULL)

    def extract_all_blocks(self,block_save_dir):
        self._save_model_frame(block_save_dir)
        # exit(0)
        for i, block_id in enumerate(self.get_blocks_id()):
            for block_sparsity in self.get_blocks_sparsity()[i]:
                # 压缩特定稀疏度的特定块 [块id, 稀疏度]
                print('\033[1;32m', '--> extracting {}: {} in sparsity {}'.format(block_id, [
                    self.graph.order_to_node.get(
                        num).get_name() for num in self.blocks[int(block_id.split('-')[-1])]],
                                                                                  block_sparsity), '\033[0m')
                self._compress_single_block(block_id, block_sparsity,block_save_dir)