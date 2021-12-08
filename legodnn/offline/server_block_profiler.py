import copy
import torch
import yaml
import numpy as np
import tqdm
import os
import json

from legodnn.online.pure_runtime import PureRuntime
from legodnn.common.manager.block_manager.abstract_block_manager import AbstractBlockManager
from legodnn.common.manager.model_manager.abstract_model_manager import AbstractModelManager
from legodnn.common.utils import logger
from legodnn.common.utils.common.file import ensure_dir



class ServerBlockProfiler:
    def __init__(self, teacher_model: torch.nn.Module, 
                 block_manager: AbstractBlockManager, model_manager: AbstractModelManager, 
                 trained_blocks_dir, test_loader, dummy_input_size,
                 device):

        self._teacher_model = teacher_model
        self._block_manager = block_manager
        self._model_manager = model_manager
        self._composed_model = PureRuntime(trained_blocks_dir, block_manager, device)
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
        for blocks_sparsity in pbar:
            model_str = self._blocks_sparsity_to_strkey(blocks_sparsity)

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

    def profile_original_blocks(self):
        ensure_dir(self._teacher_model_metrics_yaml_path)
        blocks_id = self._block_manager.get_blocks_id()
        model = copy.deepcopy(self._teacher_model).to(self._device)
        
        io_activations = self._block_manager.get_io_activation_of_all_blocks(model, self._device)
        self._model_manager.dummy_forward_to_gen_mid_data(model, self._dummy_input_size, self._device)
        blocks_input_size = list(map(lambda a: list(a.input.size())[1:], io_activations))
        blocks_output_size = list(map(lambda a: list(a.output.size())[1:], io_activations))
        [io_activation.remove() for io_activation in io_activations]
        
        blocks_info = []
        for i, block_id in enumerate(blocks_id):
            raw_block = self._block_manager.get_block_from_model(model, block_id)
            raw_block_size = self._block_manager.get_block_size(raw_block)
            raw_block_flops, raw_block_param = self._block_manager.get_block_flops_and_params(raw_block, [1] + blocks_input_size[i])
            
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
        for s in self._block_manager.get_blocks_sparsity():
            max_block_sparsity_len = max(max_block_sparsity_len, len(s))

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
        envir_sparsities = self._generate_envir_sparsities()
        envir_num = len(envir_sparsities)

        blocks_sparsities = []
        for block_index in range(blocks_num):
            for sparsity in self._block_manager.get_blocks_sparsity()[block_index]:
                for envir_sparsity in envir_sparsities:
                    baseline = copy.deepcopy(envir_sparsity)
                    # use -1 to represent the block from self._teacher_model
                    baseline[block_index] = -1

                    test_envir = copy.deepcopy(envir_sparsity)
                    test_envir[block_index] = sparsity

                    blocks_sparsities += [baseline, test_envir]

        logger.info('profile blocks acc drop')
        res_cache = self._analysis_composed_models_acc(blocks_sparsities)

        acc_drops = []
        infer_time_rel_drops = []
        model_size_drops = []
        flops_drops = []
        param_drops = []
        
        for block_index in range(blocks_num):
            block_id = self._block_manager.get_blocks_id()[block_index]
            cur_block_input_size = [1] + self._teacher_model_metrics['blocks_info'][block_index]['input_size']

            for sparsity in self._block_manager.get_blocks_sparsity()[block_index]:
                avg_acc_drop = 0.0
                infer_time_rel_drop = 0.0
                block_size_drop = 0.0
                flops_drop = 0.0
                param_drop = 0.0
                
                for envir_sparsity in envir_sparsities:
                    baseline = copy.deepcopy(envir_sparsity)
                    baseline[block_index] = -1

                    test_envir = copy.deepcopy(envir_sparsity)
                    test_envir[block_index] = sparsity

                    baseline_key = self._blocks_sparsity_to_strkey(baseline)
                    test_envir_key = self._blocks_sparsity_to_strkey(test_envir)

                    avg_acc_drop += (res_cache[baseline_key][0] - res_cache[test_envir_key][0])

                avg_acc_drop /= envir_num
                logger.info('block {} (sparsity {}) acc drop: {}'.format(block_id, sparsity, avg_acc_drop))
                
                # block size, FLOPs, param
                cur_block = self._block_manager.get_block_from_file(
                    os.path.join(self._trained_blocks_dir, 
                    self._block_manager.get_block_file_name(block_id, sparsity)),
                    self._device
                )
                
                cur_block_size = self._block_manager.get_block_size(cur_block)
                cur_block_flops, cur_block_param = self._block_manager.get_block_flops_and_params(cur_block, cur_block_input_size)
                
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
        self.profile_original_blocks()
        self.profile_all_compressed_blocks()
