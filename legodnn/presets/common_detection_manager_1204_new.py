import sys
import copy

sys.path.insert(0, '../../')
from legodnn.block_detection.model_topology_extraction import LegoDNNGraph, LegoDNNNode
# from legodnn.block_detection.base_block_detection import BaseBlockDetection
# from legodnn.block_detection.base_block_detection_1121_reused_layer import BaseBlockDetection
from legodnn.block_detection.base_block_detection_1204_new import BaseBlockDetection
from legodnn.block_detection.model_topology_extraction import LegoDNNGraph
from legodnn.block_detection.base_block_detection import COMPRESSED_LAYERS

class CommonDetectionManager:
    def __init__(self, model_graph: LegoDNNGraph, max_ratio=0.25) -> None:
        self.graph = model_graph
        self.block_detection = BaseBlockDetection(model_graph, max_ratio)

    def detection_all_blocks(self):
        self.block_detection.detection_all_block()

    def print_all_blocks(self):
        self.block_detection.print_blocks()
 
    # 返回所有用序号表示的块
    def get_blocks(self):
        return self.block_detection.blocks

    def get_num_in_block(self, block_id):
        block_idx = int(block_id.split('-')[-1])
        return self.block_detection.blocks[block_idx]

    def get_block_io_info(self, block_id):
        block_idx = int(block_id.split('-')[-1])
        return self.block_detection.blocks_start_node_is_placeholder[block_idx], self.block_detection.blocks_start_node_order[block_idx], \
                self.block_detection.blocks_end_node_order[block_idx]
                
    def get_module_name(self, node_order_or_name):
        return self.block_detection._find_module_node_in_model_name(node_order_or_name)
 
    def get_no_compressed_layers(self, block_id):
        block_idx = int(block_id.split('-')[-1])
        return self.block_detection.blocks_no_compressed_layers[block_idx]

    def get_blocks_start_node_name_hook(self, block_id):
        block_idx = int(block_id.split('-')[-1])
        return self.block_detection.blocks_start_node_name_hook[block_idx]

    def get_blocks_end_node_name_hook(self, block_id):
        block_idx = int(block_id.split('-')[-1])
        return self.block_detection.blocks_end_node_name_hook[block_idx]

    def get_blocks_start_node_hook_input_or_ouput(self, block_id):
        block_idx = int(block_id.split('-')[-1])
        return self.block_detection.blocks_start_node_hook_input_or_ouput[block_idx]

    def get_blocks_end_node_hook_input_or_ouput(self, block_id):
        block_idx = int(block_id.split('-')[-1])
        return self.block_detection.blocks_end_node_hook_input_or_ouput[block_idx]

    def get_blocks_start_node_hook_index(self, block_id):
        block_idx = int(block_id.split('-')[-1])
        return self.block_detection.blocks_start_node_hook_index[block_idx]

    def get_blocks_end_node_hook_index(self, block_id):
        block_idx = int(block_id.split('-')[-1])
        return self.block_detection.blocks_end_node_hook_index[block_idx]


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