import copy
import torch
import yaml
import os
import json

from ..online.pure_runtime import PureRuntime

from ..common.manager.block_manager.abstract_block_manager import AbstractBlockManager
from ..common.manager.model_manager.abstract_model_manager import AbstractModelManager
from ..common.utils.common.log import logger
from ..common.utils.common.file import ensure_dir
from ..common.utils.common.data_record import read_yaml



class LatencyEstimator:
    def __init__(self, block_manager: AbstractBlockManager, model_manager: AbstractModelManager, 
                 trained_blocks_dir, test_sample_num, dummy_input_size,
                 device):

        self._block_manager = block_manager
        self._model_manager = model_manager
        # self._composed_model = PureRuntime(trained_blocks_dir, block_manager, device)
        self._trained_blocks_dir = trained_blocks_dir
        self._test_sample_num = test_sample_num
        self._dummy_input_size = dummy_input_size
        self._blocks_metrics_csv_path = os.path.join(trained_blocks_dir, 'edge-blocks-metrics.csv')
        self._teacher_model_metrics_yaml_path = os.path.join(trained_blocks_dir, 'edge-teacher-model-metrics.yaml')
        self._device = device

    def profile_original_blocks(self):
        ensure_dir(self._teacher_model_metrics_yaml_path)
        blocks_id = self._block_manager.get_blocks_id()
        
        server_teacher_model_metrics = read_yaml(os.path.join(self._trained_blocks_dir, 
                                                              'server-teacher-model-metrics.yaml'))
        
        blocks_info = []
        for i, block_id in enumerate(blocks_id):
            # block_input_size = [1] + server_teacher_model_metrics['blocks_info'][i]['input_size']
            block_input_size = server_teacher_model_metrics['blocks_info'][i]['input_size']  # list or tuple
            if not isinstance(block_input_size, tuple):  # 单输入
                input_data = torch.rand([1] + block_input_size).to(self._device)
            else:  # 多输入
                input_data = ()
                for tensor_size in block_input_size:
                    input_data = input_data + (torch.rand([1] + tensor_size).to(self._device), )
            input_data = (input_data, )
            
            raw_block = self._block_manager.get_block_from_file(
                os.path.join(self._trained_blocks_dir, 
                self._block_manager.get_block_file_name(block_id, 0.0)),
                self._device
            )
            # print(111111111111)
            # print(raw_block._has_placeholder)
            # raw_block_latency = self._block_manager.get_block_latency(raw_block, self._test_sample_num, block_input_size, self._device)
            raw_block_latency = self._block_manager.get_block_latency(raw_block, self._test_sample_num, input_data, self._device)
            # print(22222222222222)
            block_info = {
                'index': i,
                'id': block_id,
                'latency': raw_block_latency
            }
            
            blocks_info += [block_info]
            logger.info('raw block info: {}'.format(json.dumps(block_info)))

        pure_runtime = PureRuntime(self._trained_blocks_dir, self._block_manager, self._device)
        pure_runtime.load_blocks([0.0 for _ in range(len(blocks_id))])
        teacher_model = pure_runtime.get_model()
        latency = self._model_manager.get_model_latency(teacher_model, self._test_sample_num, 
                                                        self._dummy_input_size, self._device)

        obj = {
            'latency': latency,
            'blocks_info': blocks_info
        }
        
        self._teacher_model_metrics = obj

        with open(self._teacher_model_metrics_yaml_path, 'w') as f:
            yaml.dump(obj, f)

    def profile_all_compressed_blocks(self):
        blocks_id = self._block_manager.get_blocks_id()
        blocks_num = len(blocks_id)
        
        server_teacher_model_metrics = read_yaml(os.path.join(self._trained_blocks_dir, 
                                                              'server-teacher-model-metrics.yaml'))

        latency_rel_drops = []
        
        for block_index in range(blocks_num):
            block_id = self._block_manager.get_blocks_id()[block_index]
            cur_raw_block = cur_block = self._block_manager.get_block_from_file(
                os.path.join(self._trained_blocks_dir, 
                self._block_manager.get_block_file_name(block_id, 0.0)),
                self._device
            )
            # cur_block_input_size = [1] + server_teacher_model_metrics['blocks_info'][block_index]['input_size']
            cur_block_input_size = server_teacher_model_metrics['blocks_info'][block_index]['input_size']  # list or tuple
            if not isinstance(cur_block_input_size, tuple):  # 单输入
                input_data = torch.rand([1] + cur_block_input_size).to(self._device)
            else:  # 多输入
                input_data = ()
                for tensor_size in cur_block_input_size:
                    input_data = input_data + (torch.rand([1] + tensor_size).to(self._device), )
            input_data = (input_data, )
            
            # cur_raw_block_latency = self._block_manager.get_block_latency(cur_raw_block, self._test_sample_num, cur_block_input_size, self._device)
            cur_raw_block_latency = self._block_manager.get_block_latency(cur_raw_block, self._test_sample_num, input_data, self._device)

            for sparsity in self._block_manager.get_blocks_sparsity()[block_index]:
                cur_block = self._block_manager.get_block_from_file(
                    os.path.join(self._trained_blocks_dir, 
                    self._block_manager.get_block_file_name(block_id, sparsity)),
                    self._device
                )
                # cur_block_latency = self._block_manager.get_block_latency(cur_block, self._test_sample_num, cur_block_input_size, self._device)
                cur_block_latency = self._block_manager.get_block_latency(cur_block, self._test_sample_num, input_data, self._device)
                
                if sparsity == 0.0:
                    cur_block_latency = cur_raw_block_latency
                latency_rel_drop = (cur_raw_block_latency - cur_block_latency) / cur_raw_block_latency

                logger.info('block {} (sparsity {}) latency rel drop: {:.3f}% '
                            '({:.3f}s -> {:.3f}s)'.format(block_id, sparsity, latency_rel_drop * 100, 
                                                          cur_raw_block_latency, cur_block_latency))
                    
                latency_rel_drops += [latency_rel_drop]
                
        ensure_dir(self._blocks_metrics_csv_path)
        csv_file = open(self._blocks_metrics_csv_path, 'w')
        csv_file.write('block_index,block_sparsity,'
                       'test_accuracy_drop,inference_time_rel_drop,model_size_drop,FLOPs_drop,param_drop\n')
        i = 0
        for block_index in range(blocks_num):
            for sparsity in self._block_manager.get_blocks_sparsity()[block_index]:
                csv_file.write('{},{},{},{},{},{},{}\n'.format(block_index, sparsity,
                                                               0, latency_rel_drops[i], 0, 0, 0))
                i += 1
        
        csv_file.close()

    def profile_all_blocks(self):
        self.profile_original_blocks()
        self.profile_all_compressed_blocks()
