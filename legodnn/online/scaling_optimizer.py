import torch
import pulp
import numpy as np
import pandas as pd
import yaml
import os
import time

from legodnn.common.manager.block_manager.abstract_block_manager import AbstractBlockManager
from legodnn.common.manager.model_manager.abstract_model_manager import AbstractModelManager
from legodnn.online.pure_runtime import PureRuntime
from legodnn.common.utils import logger


class ScalingOptimizer:
    def __init__(self, 
                 trained_blocks_dir, model_input_size,
                 block_manager: AbstractBlockManager, model_manager: AbstractModelManager,
                 device, pulp_solver=pulp.PULP_CBC_CMD(msg=False, gapAbs=0)):

        logger.info('init adaptive model runtime')
        self._block_manager = block_manager
        self._model_manager = model_manager

        self._trained_blocks_dir = trained_blocks_dir
        self._model_input_size = model_input_size
        self._device = device
        self._pulp_solver = pulp_solver

        self._load_block_metrics(os.path.join(trained_blocks_dir, 'server-blocks-metrics.csv'), 
                                 os.path.join(trained_blocks_dir, 'edge-blocks-metrics.csv'))
        self._load_model_metrics(os.path.join(trained_blocks_dir, 'server-teacher-model-metrics.yaml'), 
                                 os.path.join(trained_blocks_dir, 'edge-teacher-model-metrics.yaml'))

        self._xor_variable_id = 0
        self._last_selection = None
        self._selections_info = []
        self._before_first_adaption = True
        self._cur_original_model_infer_time = None

        self._pure_runtime = PureRuntime(trained_blocks_dir, block_manager, device)
        self._init_model()

        self._cur_blocks_infer_time = None

    def infer(self, batch_data):
        is_add_hooks = False
        hooks = []
        model = self._pure_runtime.get_model()
        if self._cur_blocks_infer_time is None:
            hooks = self._block_manager.get_time_profilers_of_all_blocks(model, self._device)
            is_add_hooks = True
            pass

        model.eval()
        
        if self._device == 'cpu':
            with torch.no_grad():
                infer_start = time.time()
                model = model.to(self._device)
                self._model_manager.forward_to_gen_mid_data(model, batch_data, self._device)
                cur_model_infer_time = time.time() - infer_start
                logger.info('infer time of current model: {}'.format(cur_model_infer_time))
        else:
            with torch.no_grad():
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                
                model = model.to(self._device)
                self._model_manager.forward_to_gen_mid_data(model, batch_data, self._device)
                
                e.record()
                torch.cuda.synchronize()
                cur_model_infer_time = s.elapsed_time(e) / 1000.
                logger.info('infer time of current model: {}'.format(cur_model_infer_time))

        if is_add_hooks:
            self._cur_blocks_infer_time = []
            for hook in hooks:
                self._cur_blocks_infer_time += [hook.infer_time]
                hook.remove()
            
            logger.info('infer time of current blocks: {}'.format(self._cur_blocks_infer_time))

            avg_ratio_list = []
            
            for block_i in range(len(self._original_blocks_infer_time)):
                biggest_block_of_cur_sparsity_cur_infer_time = self._cur_blocks_infer_time[block_i]
                biggest_block_original_infer_time = self._original_blocks_infer_time[block_i]

                i = 0
                for s in self._block_manager.get_blocks_sparsity()[: block_i]:
                    i += len(s)
                t = self._last_selection[i:]
                for j in t:
                    if j == 1:
                        break
                    i += 1

                biggest_block_of_cur_sparsity_cal_infer_time = biggest_block_original_infer_time * (1. - self._infer_time_rel_drops[i])
                ratio = biggest_block_of_cur_sparsity_cal_infer_time / biggest_block_of_cur_sparsity_cur_infer_time
                logger.info('ratio: cal infer time / real infer time of biggest block of cur sparsity: '
                            '{:.3f} / {:.3f} = {:.3f}'.format(biggest_block_of_cur_sparsity_cal_infer_time, 
                            biggest_block_of_cur_sparsity_cur_infer_time, ratio))
                
                avg_ratio_list += [ratio]
            
            avg_ratio_list = np.array(avg_ratio_list)
            std = np.std(avg_ratio_list)
            mean = np.average(avg_ratio_list)
            legal_indexes = np.where(np.abs(avg_ratio_list - mean) < std)
            logger.info('legal avg ratio indexes: {}'.format(legal_indexes))
            avg_ratio = np.average(avg_ratio_list[legal_indexes])
            logger.info('avg ratio: {}'.format(avg_ratio))
            
            self._cur_original_model_infer_time = self._original_model_infer_time / avg_ratio
            logger.info('cur cal infer time of original model: {}'.format(self._cur_original_model_infer_time))

    def _load_block_metrics(self, server_block_metrics_csv_path, edge_block_metrics_csv_file_path):
        server_block_metrics_data = pd.read_csv(server_block_metrics_csv_path)
        server_block_metrics_data = np.asarray(server_block_metrics_data)
        edge_block_metrics_data = pd.read_csv(edge_block_metrics_csv_file_path)
        edge_block_metrics_data = np.asarray(edge_block_metrics_data)

        self._acc_drops, self._infer_time_rel_drops, self._mem_drops, self._flops_drops, \
            self._param_drops = [], [], [], [], []

        total_block_index = 0
        for block_index in range(len(self._block_manager.get_blocks_sparsity())):
            this_block_num = len(self._block_manager.get_blocks_sparsity()[block_index])
            
            self._acc_drops += list(server_block_metrics_data[total_block_index: total_block_index + this_block_num, 2])
            self._infer_time_rel_drops += list(edge_block_metrics_data[total_block_index: total_block_index + this_block_num, 3])
            self._mem_drops += list(server_block_metrics_data[total_block_index: total_block_index + this_block_num, 4])
            self._flops_drops += list(server_block_metrics_data[total_block_index: total_block_index + this_block_num, 5])
            self._param_drops += list(server_block_metrics_data[total_block_index: total_block_index + this_block_num, 6])
            
            total_block_index += this_block_num

        self._acc_drops, self._infer_time_rel_drops, \
        self._mem_drops, self._flops_drops, self._param_drops = np.asarray(self._acc_drops), \
            np.asarray(self._infer_time_rel_drops), \
            np.asarray(self._mem_drops), np.asarray(self._flops_drops), np.asarray(self._param_drops)

        logger.info('load blocks metrics')

    def _load_model_metrics(self, server_model_metrics_yaml_file_path, edge_model_metrics_yaml_file_path):
        server_f = open(server_model_metrics_yaml_file_path, 'r')
        server_original_model_metrics = yaml.load(server_f, yaml.Loader)
        edge_f = open(edge_model_metrics_yaml_file_path, 'r')
        edge_original_model_metrics = yaml.load(edge_f, yaml.Loader)

        self._original_model_acc = server_original_model_metrics['test_accuracy']
        self._original_model_size = server_original_model_metrics['model_size']
        self._original_model_flops = server_original_model_metrics['FLOPs']
        self._original_model_param = server_original_model_metrics['param']
        self._original_blocks_size = np.array(list(map(lambda i: i['size'], server_original_model_metrics['blocks_info'])))
        
        self._original_blocks_infer_time = np.array(list(map(lambda i: i['latency'], edge_original_model_metrics['blocks_info'])))
        self._original_model_infer_time = edge_original_model_metrics['latency']
        
        self._blocks_input_size = list(map(lambda i: i['input_size'], server_original_model_metrics['blocks_info']))

        blocks_sparsity = self._block_manager.get_blocks_sparsity()
        original_blocks_size = []
        for i in range(len(blocks_sparsity)):
            original_blocks_size += [self._original_blocks_size[i] for _ in range(len(blocks_sparsity[i]))]
        original_blocks_size = np.asarray(original_blocks_size)
        self._blocks_size = original_blocks_size - self._mem_drops
        
        server_f.close()
        edge_f.close()

        logger.info('load model metrics')

    def _init_model(self):
        sparsest_selection = np.zeros_like(self._mem_drops)

        i = 0
        blocks_sparsity = self._block_manager.get_blocks_sparsity()
        for block_index in range(len(blocks_sparsity)):
            sparsest_selection[i + len(blocks_sparsity[block_index]) - 1] = 1
            i += len(blocks_sparsity[block_index])

        logger.info('load sparest blocks for initializing model')
        self._apply_selection_to_model(sparsest_selection)
        self._least_model_size = self._model_manager.get_model_size(self._pure_runtime.get_model())
        self._last_selection = sparsest_selection

    def _get_readable_block_selection(self, selection):
        chosen_blocks_sparsity = []
        flatten_blocks_sparsity = []
        for s in self._block_manager.get_blocks_sparsity():
            flatten_blocks_sparsity += s

        for item, s in zip(selection, flatten_blocks_sparsity):
            if item > 0:
                chosen_blocks_sparsity += [s]

        return chosen_blocks_sparsity

    def _apply_selection_to_model(self, cur_selection):
        def get_adaption_swap_mem_cost():
            if self._before_first_adaption:
                return 0
            return np.sum(np.logical_xor(self._last_selection, cur_selection) * self._blocks_size.reshape(-1))

        adaption_swap_mem_cost = get_adaption_swap_mem_cost()
        start = time.time()

        chosen_blocks_sparsity = self._get_readable_block_selection(cur_selection)
        self._pure_runtime.load_blocks(chosen_blocks_sparsity)

        adaption_time_cost = time.time() - start
        return adaption_swap_mem_cost, adaption_time_cost

    def _lp_solve(self, obj, constraints):
        prob = pulp.LpProblem('block-selection-optimization', pulp.LpMinimize)

        prob += obj
        for con in constraints:
            prob += con

        logger.info('solving...')
        status = prob.solve(self._pulp_solver)
        logger.info('solving finished')

        if status != 1:
            return None
        else:
            selection = prob.variables()[:]
            selection = list(filter(lambda v: not v.name.startswith('xor') and not v.name.startswith('__dummy'), selection))
            selection.sort(key=lambda v: int(v.name))
            return [v.varValue.real for v in selection]

    def _get_blocks_cur_infer_time(self):
        self.infer((torch.rand(self._model_input_size).to(self._device), None))

    def _get_cur_infer_time_abs_drops(self):
        if self._cur_blocks_infer_time is None:
            logger.info('no blocks infer time info, profile it through an inference')
            self._get_blocks_cur_infer_time()

        last_selection_index = []
        for i, item in enumerate(self._last_selection):
            if item > 0:
                last_selection_index += [i]

        ratio = 1. - self._infer_time_rel_drops[last_selection_index]
        baseline_infer_times = self._cur_blocks_infer_time / ratio

        logger.info('cur original blocks pred infer time: {}'.format(baseline_infer_times))

        all_baseline_infer_times = []
        blocks_sparsity = self._block_manager.get_blocks_sparsity()
        for i in range(len(blocks_sparsity)):
            all_baseline_infer_times += [baseline_infer_times[i] for _ in range(len(blocks_sparsity[i]))]
        all_baseline_infer_times = np.asarray(all_baseline_infer_times)

        return all_baseline_infer_times * self._infer_time_rel_drops

    def update_model(self, cur_max_infer_time, cur_max_model_size):
        logger.info('cur max inference time: {:.6f}s, '
                    'cur available max memory: {}B ({:.3f}MB), '
                    'try to adapt blocks'.format(cur_max_infer_time, cur_max_model_size,
                                                 cur_max_model_size / 1024 ** 2))
        
        def apply(selection, drop):
            return pulp.lpDot(selection, drop)

        def solve(cur_max_infer_time, cur_max_model_size):
            variable_num = len(self._mem_drops)
            variables = [pulp.LpVariable('{}'.format(i), lowBound=0, upBound=1, cat=pulp.LpBinary)
                         for i in range(variable_num)]

            act_acc_drop = apply(variables, self._acc_drops)
            obj = apply(variables, self._acc_drops)
            constraints = []

            act_mem_drop = apply(variables, self._mem_drops)
            act_infer_time_drop = apply(variables, self._get_cur_infer_time_abs_drops())
            act_flops_drop = apply(variables, self._flops_drops)
            constraints += [
                act_infer_time_drop >= self._cur_original_model_infer_time - cur_max_infer_time,
                act_mem_drop >= self._original_model_size - cur_max_model_size
            ]

            variable_index = 0
            for i in range(len(self._block_manager.get_blocks_sparsity())):
                n = len(self._block_manager.get_blocks_sparsity()[i])
                constraints += [pulp.lpSum(variables[variable_index: variable_index + n]) == 1]
                variable_index += n

            selection = self._lp_solve(obj, constraints)
            return selection, act_acc_drop, act_infer_time_drop, act_mem_drop, act_flops_drop

        selection, act_acc_drop, act_infer_time_drop, act_mem_drop, act_flops_drop = solve(cur_max_infer_time, cur_max_model_size)

        is_relaxed = False

        # recurse to find a feasible solution
        while selection is None and self._original_model_size > cur_max_model_size:
            is_relaxed = True
            cur_max_model_size += 0.1 * 1024 ** 2
            if cur_max_model_size < self._least_model_size:
                cur_max_model_size = self._least_model_size

            logger.info('no solution found, relax the memory constraint '
                        'to {}B ({:.3f}MB) and continue finding solution'.format(cur_max_model_size,
                                                                                 cur_max_model_size / 1024 ** 2))
            selection, act_acc_drop, act_infer_time_drop, act_mem_drop, act_flops_drop = solve(cur_max_infer_time, cur_max_model_size)
        while selection is None and self._cur_original_model_infer_time > cur_max_infer_time:
            is_relaxed = True
            cur_max_infer_time += 0.1

            logger.info('no solution found, relax the time constraint to {:.6f}s and '
                        'continue finding solution'.format(cur_max_infer_time))
            selection, act_acc_drop, act_infer_time_drop, act_mem_drop, act_flops_drop = solve(cur_max_infer_time, cur_max_model_size)

        self._cur_blocks_infer_time = None
        block_adaption_mem_swap, block_adaption_time = self._apply_selection_to_model(selection)
        self._last_selection = selection
        self._before_first_adaption = False

        # more readable and easier to use
        selection = self._get_readable_block_selection(selection)

        selection_info = {
            'blocks_sparsity': selection,

            'esti_test_accuracy': self._original_model_acc - pulp.value(act_acc_drop),
            'esti_latency': self._cur_original_model_infer_time - pulp.value(act_infer_time_drop),
            'model_size': self._original_model_size - pulp.value(act_mem_drop),
            'FLOPs': self._original_model_flops - pulp.value(act_flops_drop),
            'update_swap_mem_cost': block_adaption_mem_swap,
            'update_swap_time_cost': block_adaption_time,

            'is_relaxed': is_relaxed
        }

        self._selections_info += [selection_info]
        return selection_info

    def get_selections_info(self):
        return self._selections_info
