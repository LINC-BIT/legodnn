from concurrent.futures.thread import ThreadPoolExecutor
import copy
import torch
import yaml
import numpy as np
import tqdm
import os
import json


from ..common.manager.block_manager.abstract_block_manager import AbstractBlockManager
from ..common.manager.model_manager.abstract_model_manager import AbstractModelManager
from ..common.utils.common.log import logger
from ..common.utils.common.file import ensure_dir

class BlockProfiler:
    def __init__(self, teacher_model: torch.nn.Module, 
                 block_manager: AbstractBlockManager, model_manager: AbstractModelManager, 
                 trained_blocks_dir, test_loader, dummy_input_size,
                 device):

        self._teacher_model = teacher_model
        self._block_manager = block_manager
        self._model_manager = model_manager
        # self._composed_model = PureRuntime(trained_blocks_dir, block_manager, device)
        self._trained_blocks_dir = trained_blocks_dir
        self._dummy_input_size = dummy_input_size
        self._tested_model_metrics_csv_path = os.path.join(trained_blocks_dir, 'server-tested-models-metrics.csv')
        self._blocks_metrics_csv_path = os.path.join(trained_blocks_dir, 'server-blocks-metrics.csv')
        self._teacher_model_metrics_yaml_path = os.path.join(trained_blocks_dir, 'server-teacher-model-metrics.yaml')
        self._test_loader = test_loader
        self._device = device

    def _get_model_metrics(self, model, model_name):
        acc = self._model_manager.get_model_acc(model, self._test_loader, self._device)
        # logger.info('model ({}) test accuracy: {:.6f}'.format(model_name, acc))
        return (acc, )

    def _profile_composed_model(self, blocks_sparsity):
        model = copy.deepcopy(self._teacher_model)
        blocks_id = self._block_manager.get_blocks_id()

        for i, block_sparsity in enumerate(blocks_sparsity):
            block_id = blocks_id[i]

            if block_sparsity != -1:
                block_file_path = os.path.join(self._trained_blocks_dir,
                                               self._block_manager.get_block_file_name(block_id, block_sparsity))
                block = self._block_manager.get_block_from_file(block_file_path, self._device)

                self._block_manager.set_block_to_model(model, block_id, block)

        metrics = self._get_model_metrics(model, self._blocks_sparsity_to_strkey(blocks_sparsity))
        return metrics

    def _blocks_sparsity_to_strkey(self, blocks_sparsity):
        return '-'.join(map(lambda i: str(i).split('.')[-1], blocks_sparsity))

    def _analysis_composed_models_acc(self, blocks_sparsities):
        res_cache = {}
        
        pbar = tqdm.tqdm(blocks_sparsities, total=len(blocks_sparsities))
        # 对于每个baseline和test 
        for blocks_sparsity in pbar:
            model_str = self._blocks_sparsity_to_strkey(blocks_sparsity)
            # print(model_str)
            # 得到metrics放到res_cache里
            if model_str in res_cache.keys():
                metrics = res_cache[model_str]
                logger.info('get {} metrics in cache'.format(model_str))
            else:
                metrics = self._profile_composed_model(blocks_sparsity)
                res_cache[model_str] = np.asarray(metrics, dtype=float)
                
            pbar.set_description('model {} acc: {:.6f}'.format(model_str, metrics[0]))

        ensure_dir(self._tested_model_metrics_csv_path)
        csv_file = open(self._tested_model_metrics_csv_path, 'w')
        csv_file.write('model,test_accuracy\n')
        for model_str in res_cache.keys():
            metrics = res_cache[model_str]
            acc,  = metrics
            csv_line = '{},{}'.format(model_str, acc)
            csv_file.write(csv_line + '\n')
        csv_file.close()

        return res_cache

    # 记录teacher_model的准确率 大小 FLOPs等信息
    def profile_original_blocks(self):
        ensure_dir(self._teacher_model_metrics_yaml_path)
        blocks_id = self._block_manager.get_blocks_id()  # 块id列表
        model = copy.deepcopy(self._teacher_model).to(self._device)
        
        io_activations = self._block_manager.get_io_activation_of_all_blocks(model, self._device)
        self._model_manager.dummy_forward_to_gen_mid_data(model, self._dummy_input_size, self._device)
        
        blocks_input_size = []
        blocks_output_size = []
        for block_index, block_id in enumerate(blocks_id):
            print("当前块的编号: {}".format(block_id))
            detection_manager = self._block_manager.detection_manager
            # 得到输入数据
            input_layer_activation_list = []
            start_module_name_list = detection_manager.get_blocks_start_node_name_hook(block_id)      
            print("当前需要钩的输入节点: {}".format(start_module_name_list))
            for start_module_name in start_module_name_list:
                input_layer_activation_list.append(io_activations.get(start_module_name))
            input_need_hook_input_list = []
            start_node_hook_input_or_ouput_list = detection_manager.get_blocks_start_node_hook_input_or_ouput(block_id)
            for start_node_hook_input_or_ouput in start_node_hook_input_or_ouput_list:
                input_need_hook_input_list.append(start_node_hook_input_or_ouput == 0)
            start_hook_index_list = detection_manager.get_blocks_start_node_hook_index(block_id)
            input_data = ()
            # print("当前输入在钩出数据中的位置: {}".format(start_hook_index_list))
            for i, layer_activation in enumerate(input_layer_activation_list):
                print("输入节点{}，钩出输入的长度{}，索引位置{}".format(start_module_name_list[i], len(layer_activation.input_list), start_hook_index_list[i]))
                # print("钩出数据的类型{}".format(type(layer_activation.input_list[start_hook_index_list[i]] if input_need_hook_input_list[i] else layer_activation.output_list[start_hook_index_list[i]])))
                input_data = input_data + (layer_activation.input_list[start_hook_index_list[i]] if input_need_hook_input_list[i] else layer_activation.output_list[start_hook_index_list[i]],)
            # if len(start_module_name_list) == 1:
            #     input_data = input_data[0]
            
            # 得到输出数据
            output_layer_activation_list = []
            end_module_name_list = detection_manager.get_blocks_end_node_name_hook(block_id)
            for end_module_name in end_module_name_list:
                output_layer_activation_list.append(io_activations.get(end_module_name))
            output_need_hook_input_list = []
            end_node_hook_input_or_ouput_list = detection_manager.get_blocks_end_node_hook_input_or_ouput(block_id)
            for end_node_hook_input_or_ouput in end_node_hook_input_or_ouput_list:
                output_need_hook_input_list.append(end_node_hook_input_or_ouput == 0)
            end_hook_index_list = detection_manager.get_blocks_end_node_hook_index(block_id)
            output_data = ()
            for i, layer_activation in enumerate(output_layer_activation_list):
                output_data = output_data + (layer_activation.output_list[end_hook_index_list[i]] if output_need_hook_input_list[i] else layer_activation.output_list[end_hook_index_list[i]],)
            # if len(end_module_name_list) == 1:
            #     output_data = output_data[0]
            
            # blocks_input_size.append(list(input_data.size())[1:])
            # blocks_output_size.append(list(output_data.size())[1:])
            
            block_input_size = ()
            # print(len(start_module_name_list), len(input_data))
            # if len(start_module_name_list) > 1:
            #     print(input_data)
            for input_tensor in input_data:
                block_input_size = block_input_size + (list(input_tensor.size())[1:], )
            if len(start_module_name_list) == 1:
                block_input_size = block_input_size[0]
            blocks_input_size.append(block_input_size)  # list or tuple
            
            block_output_size = ()
            for output_tensor in output_data:
                block_output_size = block_output_size + (list(output_tensor.size())[1:], )
            if len(end_module_name_list) == 1:
                block_output_size = block_output_size[0]
            blocks_output_size.append(block_output_size)

        self._block_manager.remove_io_activations(io_activations)

        # model.eval()
        # for i in range(10000):
        #     input = torch.ones(self._dummy_input_size).cuda()
        #     model(input)
        blocks_info = []
        for i, block_id in enumerate(blocks_id):
            raw_block = self._block_manager.get_block_from_model(model, block_id)
            raw_block_size = self._block_manager.get_block_size(raw_block)
            
            if not isinstance(blocks_input_size[i], tuple):  # 单输入
                input_data = torch.rand([1] + blocks_input_size[i]).to(self._device)
            else:  # 多输入
                input_data = ()
                for tensor_size in blocks_input_size[i]:
                    input_data = input_data + (torch.rand([1] + tensor_size).to(self._device), )
            input_data = (input_data, )
            
            # raw_block_flops, raw_block_param = self._block_manager.get_block_flops_and_params(raw_block, [1] + blocks_input_size[i])
            raw_block_flops, raw_block_param = self._block_manager.get_block_flops_and_params(raw_block, input_data)
            
            block_info = {
                'index': i,
                'id': block_id,
                'size': raw_block_size,
                'FLOPs': raw_block_flops,
                'param': raw_block_param,
                'input_size': blocks_input_size[i],
                'output_size': blocks_output_size[i]
            }
            
            blocks_info += [block_info]
            logger.info('raw block info: {}'.format(json.dumps(block_info)))

        acc = self._model_manager.get_model_acc(model, self._test_loader, self._device)
        model_size = self._model_manager.get_model_size(model)
        flops, param = self._model_manager.get_model_flops_and_param(model, self._dummy_input_size)

        obj = {
            'test_accuracy': acc,
            'model_size': model_size,
            'FLOPs': flops,
            'param': param,
            'blocks_info': blocks_info
        }
        
        self._teacher_model_metrics = obj

        with open(self._teacher_model_metrics_yaml_path, 'w') as f:
            yaml.dump(obj, f)

    def _generate_envir_sparsities(self):
        max_block_sparsity_len = -1
        # 对于每个块的稀疏度列表
        for s in self._block_manager.get_blocks_sparsity():
            max_block_sparsity_len = max(max_block_sparsity_len, len(s))
        # 确定本次最多的稀疏度个数
        block_num = len(self._block_manager.get_blocks_sparsity())
        # use -1 to represent the block from self._teacher_model
        res = [[-1 for _ in range(block_num)]]

        for j in range(max_block_sparsity_len):
            tmp = []
            for k in range(block_num):
                si = min(len(self._block_manager.get_blocks_sparsity()[k]) - 1, j)
                tmp += [self._block_manager.get_blocks_sparsity()[k][si]]
            res += [tmp]
        
        # reduce profile time cost
        num = len(res)
        if num <= 3:
            return res
        
        a, b, c = 0, (num - 1) // 2, num - 1

        return [res[a], res[b], res[c]]

    def profile_all_compressed_blocks(self):
        blocks_id = self._block_manager.get_blocks_id()
        blocks_num = len(blocks_id)
        # 得3种 块的稀获疏度列表  形状为: 3*块数 作为训练环境（模型稀疏度）
        envir_sparsities = self._generate_envir_sparsities()
        envir_num = len(envir_sparsities)

        blocks_sparsities = []
        # 对于每个块
        for block_index in range(blocks_num):
            # 这个块的一个稀疏度
            for sparsity in self._block_manager.get_blocks_sparsity()[block_index]:
                # 对于每个 块稀疏度列表
                for envir_sparsity in envir_sparsities:
                    # 将这个块的位置的稀疏度换为-1作为baseline
                    baseline = copy.deepcopy(envir_sparsity)
                    # use -1 to represent the block from self._teacher_model
                    baseline[block_index] = -1
                    # 将这个块的位置的稀疏度换为特定稀疏度作为测试
                    test_envir = copy.deepcopy(envir_sparsity)
                    test_envir[block_index] = sparsity
                    # 打包
                    blocks_sparsities += [baseline, test_envir]

        logger.info('profile blocks acc drop')
        # 评估不同稀疏度的不同块在不同模型稀疏度下的训练结果
        res_cache = self._analysis_composed_models_acc(blocks_sparsities)

        acc_drops = []
        infer_time_rel_drops = []
        model_size_drops = []
        flops_drops = []
        param_drops = []
        
        # 对于每个块
        for block_index in range(blocks_num):
            # 得到块id和它的输入形状
            block_id = self._block_manager.get_blocks_id()[block_index]
            cur_block_input_size = self._teacher_model_metrics['blocks_info'][block_index]['input_size']
            if not isinstance(cur_block_input_size, tuple):  # 单输入
                input_data = torch.rand([1] + cur_block_input_size).to(self._device)
            else:  # 多输入
                input_data = ()
                for tensor_size in cur_block_input_size:
                    input_data = input_data + (torch.rand([1] + tensor_size).to(self._device), )
            input_data = (input_data, )
            # 对于这个块的每一个稀疏度
            for sparsity in self._block_manager.get_blocks_sparsity()[block_index]:
                avg_acc_drop = 0.0
                infer_time_rel_drop = 0.0
                block_size_drop = 0.0
                flops_drop = 0.0
                param_drop = 0.0
                # 每个模型稀疏度
                for envir_sparsity in envir_sparsities:
                    baseline = copy.deepcopy(envir_sparsity)
                    baseline[block_index] = -1

                    test_envir = copy.deepcopy(envir_sparsity)
                    test_envir[block_index] = sparsity

                    baseline_key = self._blocks_sparsity_to_strkey(baseline)
                    test_envir_key = self._blocks_sparsity_to_strkey(test_envir)

                    avg_acc_drop += (res_cache[baseline_key][0] - res_cache[test_envir_key][0])
                # 算平均
                avg_acc_drop /= envir_num
                logger.info('block {} (sparsity {}) acc drop: {}'.format(block_id, sparsity, avg_acc_drop))
                
                # block size, FLOPs, param
                cur_block = self._block_manager.get_block_from_file(
                    os.path.join(self._trained_blocks_dir, 
                    self._block_manager.get_block_file_name(block_id, sparsity)),
                    self._device
                )
                
                cur_block_size = self._block_manager.get_block_size(cur_block)
                cur_block_flops, cur_block_param = self._block_manager.get_block_flops_and_params(cur_block, input_data)
                
                raw_block_size = self._teacher_model_metrics['blocks_info'][block_index]['size']
                raw_block_flops = self._teacher_model_metrics['blocks_info'][block_index]['FLOPs']
                raw_block_param = self._teacher_model_metrics['blocks_info'][block_index]['param']
                
                block_size_drop = raw_block_size - cur_block_size
                flops_drop = raw_block_flops - cur_block_flops
                param_drop = raw_block_param - cur_block_param
                
                if sparsity == 0.0:
                    block_size_drop = 0
                    flops_drop = 0
                    param_drop = 0

                logger.info('block {} (sparsity {}) size drop: {}B ({:.3f}MB), '
                            'FLOPs drop: {:.3f}M, param drop: {:.3f}M'.format(block_id, sparsity, 
                                                                              block_size_drop, block_size_drop / 1024**2, 
                                                                              flops_drop / 1e6, param_drop / 1e6))
                    
                acc_drops += [avg_acc_drop]
                model_size_drops += [block_size_drop]
                infer_time_rel_drops += [infer_time_rel_drop]
                flops_drops += [flops_drop]
                param_drops += [param_drop]
                
        ensure_dir(self._blocks_metrics_csv_path)
        csv_file = open(self._blocks_metrics_csv_path, 'w')
        csv_file.write('block_index,block_sparsity,'
                       'test_accuracy_drop,inference_time_rel_drop,model_size_drop,FLOPs_drop,param_drop\n')
        i = 0
        for block_index in range(blocks_num):
            for sparsity in self._block_manager.get_blocks_sparsity()[block_index]:
                csv_file.write('{},{},{},{},{},{},{}\n'.format(block_index, sparsity,
                                                               acc_drops[i], infer_time_rel_drops[i], 
                                                               model_size_drops[i], flops_drops[i],
                                                               param_drops[i]))
                i += 1
        
        csv_file.close()

    def profile_all_blocks(self):
        # 记录teacher_model的准确率 大小 FLOPs等信息
        self.profile_original_blocks()

        self.profile_all_compressed_blocks()
