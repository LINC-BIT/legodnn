from re import L
import re
import sys

from networkx.classes.function import non_edges
sys.path.insert(0, '../../')
from legodnn.block_detection.model_topology_extraction import LegoDNNGraph, LegoDNNNode
from queue import Queue
from typing import Dict, List

COMPRESSED_LAYERS = ['Conv2d', 'Linear', 'ConvTranspose2d'] # 这些是可以被压缩的层
PARAM_REUSE_LAYERS = COMPRESSED_LAYERS + ['BatchNorm2d'] # 带参数的网络层如果被重用，必须在同一个块内
BLOCK_DETECTION_MODE = ['oto', 'otm', 'mto', 'mtm']

class BlockDetectionObjectDetection:
    def __init__(self, backbone_graph=None, neck_graph=None, backbone_compress_num=(4,6)) -> None:
       if backbone_graph is not None:
           pass
       
       pass

if __name__=='__main__':
    import torch
    from legodnn.third_party.nni.common.graph_utils import build_graph, build_module_graph
    # net = resnet152().cuda()
    # data = torch.ones(1, 3, 224, 224).cuda()
    from cv_task.image_classification.cifar.models import inceptionv3, cbam_resnet18, resnet18
    net = inceptionv3().cuda()
    data = torch.ones(1, 3, 32, 32).cuda()

    print(net)
    # 通过nni建图
    module_graph = build_module_graph(net, data)
    name_to_node = module_graph.name_to_node  # dict
    input_to_node = module_graph.input_to_node  # defaultdict
    output_to_node = module_graph.output_to_node  # dict

    graph = LegoDNNGraph()
    graph.build_graph(name_to_node, input_to_node, output_to_node)
    graph.print_ordered_node()
    # exit(0)

    # block_detection = BlockDetection(graph, min_compress_num=4, max_compress_num=6) # resnet18
    block_detection = BlockDetectionObjectDetection(graph, min_compress_num=8, max_compress_num=16) # inceptionv3
    block_detection.print_reuse_layers()
    # exit(0)
    block_detection.detection_all_block()
    # print(block_detection.blocks)
    block_detection.print_blocks()
    # graph.show_graph()
