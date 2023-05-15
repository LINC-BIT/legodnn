import sys
import copy

sys.path.insert(0, '../../')
from legodnn.block_detection.model_topology_extraction import LegoDNNGraph, LegoDNNNode
from queue import Queue
from typing import Dict, List
# from legodnn.block_detection.base_block_detection import BaseBlockDetection
from legodnn.block_detection.base_block_detection_1121_two_stage import BaseBlockDetection
from legodnn.block_detection.model_topology_extraction import LegoDNNGraph
from legodnn.block_detection.base_block_detection import COMPRESSED_LAYERS

class ObjectDeteDetectionManager:
    def __init__(self, model_graph: LegoDNNGraph, detection_mode='whole', whole_compress_num_range=(4,6), detection_cfg={}) -> None:
        self.graph = model_graph
        self.detection_mode = detection_mode
        if self.detection_mode == 'whole':
            self.whole_compress_num_range = whole_compress_num_range
            self.whole_block_detection = BaseBlockDetection(model_graph, min_compress_num=whole_compress_num_range[0],
                                                     max_compress_num=whole_compress_num_range[1], mode='oto')
                                                     
        elif self.detection_mode == 'component':
            self.component_types = []
            self.component_block_detections = {}
            # 对于每个组件 由cfg获取其LegoDNNGraph和压缩层数范围 依据这些信息分别定义detection
            if detection_cfg.get('backbone'):
                self.component_types.append('backbone')
                backbone_graph = self.graph.get_subgraph('backbone')
                backbone_compress_num_range = detection_cfg.get('backbone_compress_num_range')
                backbone_mode = detection_cfg.get('backbone_mode')
                
                backbone_block_detection = BaseBlockDetection(backbone_graph, min_compress_num=backbone_compress_num_range[0], 
                                                            max_compress_num=backbone_compress_num_range[1], mode=backbone_mode)
                # backbone_block_detection.specialed_output_nodes = self._find_all_specialed_output_nodes_by_order(backbone_graph, model_graph)
                self.component_block_detections.update({'backbone': backbone_block_detection})
                
            if detection_cfg.get('neck'):
                self.component_types.append('neck')
                neck_graph = self.graph.get_subgraph('neck')
                neck_compress_num_range = detection_cfg.get('neck_compress_num_range')
                neck_mode = detection_cfg.get('neck_mode')
                neck_block_detection = BaseBlockDetection(neck_graph, min_compress_num=neck_compress_num_range[0], 
                                                        max_compress_num=neck_compress_num_range[1], mode=neck_mode)
                # neck_block_detection.specialed_output_nodes = self._find_all_specialed_output_nodes_by_order(neck_graph, model_graph)
                self.component_block_detections.update({'neck': neck_block_detection})
                
        # whole_block_detection    or    component_block_detections
        

    def _find_all_specialed_output_nodes_by_name(self, component_name, component_graph, model_graph):
        specialed_output_nodes = []
        node_name_list = component_graph.node_dict.keys()
        
        if len(component_name)>0:
            origin_node_name_list = [component_name + '.' + node_name for node_name in node_name_list]
        else:
            origin_node_name_list = node_name_list

        for node_name in node_name_list:
            if len(component_name)>0:
                origin_node_name = component_name + '.' + node_name
            else:
                origin_node_name = node_name

            model_next_node_name_list = model_graph.node_dict[origin_node_name].next_nodes.keys()

            for model_next_node_name in model_next_node_name_list:
                if model_next_node_name not in origin_node_name_list:
                    specialed_output_nodes.append(component_graph.node_dict[node_name].serial_number)
                    break

        print("backbone中特殊节点: ")
        print(list(set(specialed_output_nodes)))
        return list(set(specialed_output_nodes))

    def _find_all_specialed_output_nodes_by_order(self, component_graph, model_graph):
        specialed_output_nodes = []
        node_order_list = component_graph.order_to_node.keys()
    
        for node_order in node_order_list:
            model_next_node_name_list = model_graph.order_to_node[node_order].next_nodes.keys()

            for model_next_node_name in model_next_node_name_list:
                if  model_graph.node_dict[model_next_node_name].serial_number not in node_order_list:
                    specialed_output_nodes.append(node_order)
                    break

        print("backbone中特殊节点: ")
        print(list(set(specialed_output_nodes)))
        return list(set(specialed_output_nodes))


    def detection_all_blocks_of_components(self):
        if self.detection_mode == 'whole':
            self.whole_block_detection.detection_all_block()
        elif self.detection_mode == 'component':
            for component_type in self.component_types:
                if component_type == 'backbone':
                    self.component_block_detections['backbone'].detection_all_block()
                elif component_type == 'neck':
                    self.component_block_detections['neck'].detection_all_block()

                    # neck_detection = self.component_block_detections['neck']
                    # neck_detection.blocks.append('neck')
                    # neck_block = list(neck_detection.graph.order_to_node.keys())
                    # print(neck_block)
                    # neck_detection.blocks_no_compressed_layers.append(neck_detection._find_block_all_paths_first_compressed_layer_name_from_end_nodes(neck_block, [neck_block[-1]]))
                    # neck_detection.blocks_start_node_is_placeholder.append([[0]])
                    # neck_detection.blocks_start_node_name_hook.append(['neck'])
                    # neck_detection.blocks_start_node_hook_input_or_ouput.append([0])
                    # neck_detection.blocks_start_node_hook_index.append([0])
                    # neck_detection.blocks_end_node_name_hook.append(['neck'])
                    # neck_detection.blocks_end_node_hook_input_or_ouput.append([1])
                    # neck_detection.blocks_end_node_hook_index.append([0])

        

    def print_all_blocks_of_components(self):
        if self.detection_mode == 'whole':
            self.whole_block_detection.print_blocks()
        elif self.detection_mode == 'component':
            for component_type in self.component_types:
                if component_type == 'backbone':
                    print('backbone部分, 共有{}个块'.format(len(self.component_block_detections['backbone'].blocks)))
                    self.component_block_detections['backbone'].print_blocks()
                elif component_type == 'neck':
                    print('neck部分, 共有{}个块'.format(len(self.component_block_detections['neck'].blocks)))
                    self.component_block_detections['neck'].print_blocks()
    
    # 返回所有用序号表示的块
    def get_blocks(self):
        if self.detection_mode == 'whole':
            return self.whole_block_detection.blocks
        elif self.detection_mode == 'component':
            blocks = []
            for component_type in self.component_types:
                blocks += self.component_block_detections[component_type].blocks
            return blocks

    # def get_blocks_id(self):
    #     blocks_id = []
    #     if self.detection_mode == 'whole':
    #         for i in range(0, len(self.whole_block_detection.blocks)):
    #             blocks_id.append('whole-block-{}'.format(i))
    #     elif self.detection_mode == 'component':
    #         for component_type in self.component_types:
    #             for i in range(0, len(self.component_block_detections[component_type].blocks)):
    #                 blocks_id.append('{}-block-{}'.format(component_type, i))
    #     return blocks_id

    def _solve_block_id(self, block_id):
        block_idx = int(block_id.split('-')[-1])
        if self.detection_mode == 'whole':
            return None, block_idx
        elif self.detection_mode == 'component':
            curr_index = 0
            for component_type in self.component_types:
                block_len = len(self.component_block_detections[component_type].blocks)
                if curr_index + block_len > block_idx:
                    return component_type, block_idx - curr_index
                curr_index += block_len

    def get_num_in_block(self, block_id):
        component_type, inner_idx = self._solve_block_id(block_id)
        if not component_type:
            return self.whole_block_detection.blocks[inner_idx]
        else:
            return self.component_block_detections[component_type].blocks[inner_idx]

    def get_block_io_info(self, block_id):
        component_type, inner_idx = self._solve_block_id(block_id)
        if not component_type:
            return self.whole_block_detection.blocks_start_node_is_placeholder[inner_idx], \
                self.whole_block_detection.blocks_start_node_order[inner_idx], self.whole_block_detection.blocks_end_node_order[inner_idx]
        else:
            detection = self.component_block_detections[component_type]
            return detection.blocks_start_node_is_placeholder[inner_idx], detection.blocks_start_node_order[inner_idx], \
                detection.blocks_end_node_order[inner_idx]

    def get_module_name(self, node_order_or_name):
        if self.detection_mode == 'whole':
            return self.whole_block_detection._find_module_node_in_model_name(node_order_or_name)
        elif self.detection_mode == 'component':
            for component_type in self.component_types:
                module_name = self.component_block_detections[component_type]._find_module_node_in_model_name(node_order_or_name)
                if module_name:
                    return module_name
    
    def get_no_compressed_layers(self, block_id):
        component_type, inner_idx = self._solve_block_id(block_id)
        if not component_type:
            return self.whole_block_detection.blocks_no_compressed_layers[inner_idx]
        else:
            return self.component_block_detections[component_type].blocks_no_compressed_layers[inner_idx]

    def get_blocks_start_node_name_hook(self, block_id):
        component_type, inner_idx = self._solve_block_id(block_id)
        if not component_type:
            return self.whole_block_detection.blocks_start_node_name_hook[inner_idx]
        else:
            return self.component_block_detections[component_type].blocks_start_node_name_hook[inner_idx]

    def get_blocks_end_node_name_hook(self, block_id):
        component_type, inner_idx = self._solve_block_id(block_id)
        if not component_type:
            return self.whole_block_detection.blocks_end_node_name_hook[inner_idx]
        else:
            return self.component_block_detections[component_type].blocks_end_node_name_hook[inner_idx]

    def get_blocks_start_node_hook_input_or_ouput(self, block_id):
        component_type, inner_idx = self._solve_block_id(block_id)
        if not component_type:
            return self.whole_block_detection.blocks_start_node_hook_input_or_ouput[inner_idx]
        else:
            return self.component_block_detections[component_type].blocks_start_node_hook_input_or_ouput[inner_idx]

    def get_blocks_end_node_hook_input_or_ouput(self, block_id):
        component_type, inner_idx = self._solve_block_id(block_id)
        if not component_type:
            return self.whole_block_detection.blocks_end_node_hook_input_or_ouput[inner_idx]
        else:
            return self.component_block_detections[component_type].blocks_end_node_hook_input_or_ouput[inner_idx]

    def get_blocks_start_node_hook_index(self, block_id):
        component_type, inner_idx = self._solve_block_id(block_id)
        if not component_type:
            return self.whole_block_detection.blocks_start_node_hook_index[inner_idx]
        else:
            return self.component_block_detections[component_type].blocks_start_node_hook_index[inner_idx]

    def get_blocks_end_node_hook_index(self, block_id):
        component_type, inner_idx = self._solve_block_id(block_id)
        if not component_type:
            return self.whole_block_detection.blocks_end_node_hook_index[inner_idx]
        else:
            return self.component_block_detections[component_type].blocks_end_node_hook_index[inner_idx]

    

    # def get_block_by_id(self, model, block_id):
    #     block_component = block_id.split('-')[0] # 属于哪个组件
    #     block_index = int(block_id.split('-')[-1])  # 块号
    #     if block_component == 'neck':
    #         return copy.deepcopy(get_module(model, 'neck'))
    #     else:


    


if __name__=='__main__':
    from functools import partial

    from mmdet.apis import init_detector
    from mmdet.datasets import replace_ImageToTensor
    from mmdet.datasets.pipelines import Compose
    from legodnn.utils.dl.common.model import get_module
    import json
    import numpy as np

    config = '/data/gxy/legodnn-public-version_object_detection/cv_task/object_detection/mmdet_models/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # checkpoint = '/data/gxy/legodnn-public-version_object_detection/cv_task/object_detection/mmdet_models/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    image_path = '/data/gxy/legodnn-public-version_object_detection/cv_task/object_detection/test.jpg'


    device = 'cuda'
    detector = init_detector(config, device=device) 
    
    detector.forward = partial(detector.legodnn_jit_forward)
    detector.eval()

    from legodnn.block_detection.model_topology_extraction import topology_extraction
    model_graph = topology_extraction(detector, (1,3,300,400), device=device)
    model_graph.print_ordered_node()
    # backbone_graph = topology_extraction(get_module(detector, 'backbone'), (1,3,300,400), device=device)
    # neck_graph = topology_extraction(get_module(detector, 'neck'), ((1,256,74,100), (1,512,38,50), (1,1024,19,25), (1,2048,10,13)), device=device)
    
    detection_cfg = {
        'backbone': True,
        'backbone_compress_num_range': (6, 8),
        'neck': True,
        'neck_compress_num_range': (8, 9)
    }
    block_detection = ObjectDeteDetectionManager(model_graph, detection_mode='component', detection_cfg=detection_cfg)
    block_detection.detection_all_blocks_of_components()
    block_detection.print_all_blocks_of_components()

    # from legodnn.block_detection.block_detection import BlockDetection
    # block_detection = BlockDetection(graph, min_compress_num=6, max_compress_num=12) # resnet
    # block_detection.detection_all_block()
    # block_detection.print_blocks()