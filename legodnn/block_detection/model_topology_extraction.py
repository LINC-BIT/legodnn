from re import sub
import matplotlib as mpl
from networkx.classes.function import subgraph
mpl.use('Agg')
from os import name
import queue
import random
import copy
from tensorboard.compat.proto import graph_pb2
import torch
import torch.nn as nn
from torch.nn.modules import module
from torch.utils import data
from legodnn.third_party.nni.compression.pytorch.utils.shape_dependency import ChannelDependency
from torchvision.models import resnet18, vgg16, resnet50

import onnx
from onnx import shape_inference

from legodnn.third_party.nni.common.graph_utils import build_graph, build_module_graph, build_module_graph_with_unpack_manually

import networkx as nx
import matplotlib.pyplot as plt


class LegoDNNNode:
    def __init__(self, name, type, op_type, auxiliary):
        self._name = name
        self._type = type
        self._op_type = op_type
        self._auxiliary = auxiliary

        self.pre_nodes = {}
        self.next_nodes = {}
        self.serial_number = -1

    def get_name(self):
        return self._name

    def get_type(self):
        return self._type

    def get_op_type(self):
        return self._op_type

    def add_input(self, node):
        if not self.pre_nodes.get(node.get_name()):
            self.pre_nodes.update({node.get_name(): node})

    def add_output(self, node):
        if not self.next_nodes.get(node.get_name()):
            self.next_nodes.update({node.get_name(): node})

    def has_input(self, node=None):
        if node == None:
            return self.pre_nodes
        return self.pre_nodes.get(node.get_name())

    def has_output(self, node):
        return self.next_nodes.get(node.get_name())



class LegoDNNGraph:
    def __init__(self):
        self.node_dict = {}
        self.start_node = []  # 可能不止一个节点没有输入
        self.end_node = []
        self.order_to_node = {} # 根据序号索引节点

    def _add_node(self, node):
        if not self.node_dict.get(node.get_name()):
            self.node_dict.update({node.get_name(): node})

    def _add_edge(self, src, des):
        src.add_output(des)
        des.add_input(src)

    def _find_start_node(self):
        for _, node in self.node_dict.items():
            if not node.has_input():
                self.start_node.append(node)
            # print('节点{}前驱{}后继{}'.format(node.get_name(), list(node.pre_nodes.keys()), list(node.next_nodes.keys())))
        # for node in self.start_node:
        #     print('图的开始节点是{}'.format(node.get_name()))   

    # 给节点分配序号
    # def _assign_serial_number(self):
    #     node_queue = queue.LifoQueue()
    #     serial_number = 1
    #     for node in self.start_node:
    #         node_queue.put(node)
    #     while not node_queue.empty():
    #         curr_node = node_queue.get()
    #         curr_node.serial_number = serial_number
    #         self.order_to_node[serial_number] = curr_node
    #         serial_number += 1
    #         # 遍历当前节点的后继
    #         for name, node in curr_node.next_nodes.items():
    #             # 检查是否它每一个前驱都被编号 否则continue
    #             flag = True
    #             for pre_node in list(node.pre_nodes.values()):
    #                 if pre_node.serial_number == -1:
    #                     flag = False
    #             if not flag:
    #                 continue
    #             node_queue.put(node)
        
    def _do_assign_serial_number(self, curr_node, serial_number):
        curr_node.serial_number = serial_number
        self.order_to_node[serial_number] = curr_node
        serial_number += 1
        # 遍历当前节点的后继
        for name, node in curr_node.next_nodes.items():
            # 检查是否它每一个前驱都被编号 否则continue
            flag = True
            for pre_node in list(node.pre_nodes.values()):
                if pre_node.serial_number == -1:
                    flag = False
            if not flag or node.serial_number != -1:
                continue
            serial_number = self._do_assign_serial_number(node, serial_number)
        return serial_number

    def _assign_serial_number(self):
        serial_number = 1
        for node in self.start_node:
            serial_number = self._do_assign_serial_number(node, serial_number)


    def len(self):
        return len(self.order_to_node)
            
    def build_graph(self, name_to_node, input_to_node, output_to_node):
        # 将所有节点加入图
        for name, node in name_to_node.items():
            self._add_node(LegoDNNNode(name, node.type, node.op_type, node.auxiliary))
        # 对于图中每个节点
        for _, node in self.node_dict.items():
            # 找到其对应的原始NodeByGroup节点
            origin_node = name_to_node.get(node.get_name())
            # 如果有把node作为output的节点
            for output in origin_node.inputs:
                pre_node = output_to_node.get(output)
                if pre_node:
                    pre_node = self.node_dict.get(pre_node.unique_name)
                    self._add_edge(pre_node, node)
            #如果有把node作为input的节点列表
            for input in origin_node.outputs:
                next_node_list = input_to_node.get(input)
                if next_node_list:
                    for next_node in next_node_list:
                        next_node = self.node_dict.get(next_node.unique_name)
                        self._add_edge(node, next_node)
        self._find_start_node()
        self._assign_serial_number()
        self.end_node.append(self.order_to_node.get(len(self.order_to_node)))
        
    def build_graph_with_unpack_manually(self, name_to_node, input_to_node, output_to_node):
        # 将所有节点加入图
        for name, node in name_to_node.items():
            if node.op_type in ['prim::ListUnpack', 'prim::TupleUnpack']:
                continue
            self._add_node(LegoDNNNode(name, node.type, node.op_type, node.auxiliary))
        # 对于图中每个节点
        for _, node in self.node_dict.items():
            # 找到其对应的原始NodeByGroup节点
            origin_node = name_to_node.get(node.get_name())
            # 如果有把node作为output的节点
            for output in origin_node.inputs:
                pre_node = output_to_node.get(output)
                if pre_node:
                    pre_node = self.node_dict.get(pre_node.unique_name)
                    if pre_node:
                        self._add_edge(pre_node, node)
            #如果有把node作为input的节点列表
            for input in origin_node.outputs:
                next_node_list = input_to_node.get(input)
                if next_node_list:
                    for next_node in next_node_list:
                        next_node = self.node_dict.get(next_node.unique_name)
                        if next_node:
                            self._add_edge(node, next_node)
        self._find_start_node()
        self._assign_serial_number()
        self.end_node.append(self.order_to_node.get(len(self.order_to_node)))
        
    def print_ordered_node(self):
        for num, node in self.order_to_node.items():
            print("num {}, name {}, type {}, op_type {}, auxiliary {}, pre_nodes {}, next_nodes {}"
                .format(num, node.get_name(), node._type, node._op_type, node._auxiliary, list(node.pre_nodes.keys()), list(node.next_nodes.keys())))

    def print_start_node(self):
        print("当前图的开始节点:")
        for node in self.start_node:
            print("num {}, name {}, type {}, op_type {}, auxiliary {}, pre_nodes {}, next_nodes {}"
                .format(node.serial_number, node.get_name(), node._type, node._op_type, node._auxiliary, list(node.pre_nodes.keys()), list(node.next_nodes.keys())))

    def print_end_node(self):
        print("当前图的终止节点:")
        for node in self.end_node:
            print("num {}, name {}, type {}, op_type {}, auxiliary {}, pre_nodes {}, next_nodes {}"
                .format(node.serial_number, node.get_name(), node._type, node._op_type, node._auxiliary, list(node.pre_nodes.keys()), list(node.next_nodes.keys())))
            
    def get_subgraph(self, prefix):
        def _in_subgraph(node: LegoDNNNode):
            order = node.serial_number
            pre_node = self.order_to_node.get(order - 1)
            next_node = self.order_to_node.get(order + 1)
            if pre_node and next_node:
                if pre_node.get_name().startswith(prefix) and next_node.get_name().startswith(prefix):
                    return True
            return False
        
        original_start_node_name = []
        for node in self.start_node:
            original_start_node_name.append(node.get_name())

        subgraph_node_dict = {}
        subgraph_order_to_node = {}
        start_node = []
        end_node = []

        auxiliary_start_node_name = set()
        auxiliary_end_node_name = set()

        # 得到子图中的节点
        auxiliary_order_to_node = {}

        for order, node in self.order_to_node.items():
            name = node.get_name()
            # 若非子图中节点
            if not name.startswith(prefix) and not _in_subgraph(node):
                continue
            # 若是子图中节点 查看其前驱和后继 若不在子图中 则为开始节点或结束节点
            auxiliary_pre_nodes = {}
            for pre_name, pre_node in node.pre_nodes.items():
                if pre_name.startswith(prefix) or _in_subgraph(pre_node):
                    auxiliary_pre_nodes.update({pre_name: None})
                else:
                    auxiliary_start_node_name.add(name)
            auxiliary_next_nodes = {}
            for next_name, next_node in node.next_nodes.items():
                if next_name.startswith(prefix) or _in_subgraph(next_node):
                    auxiliary_next_nodes.update({next_name: None})
                else:
                    auxiliary_end_node_name.add(name)
            
            auxiliary_node = LegoDNNNode(node.get_name(), node.get_type(), node.get_op_type(), node._auxiliary)
            auxiliary_node.serial_number = node.serial_number
            auxiliary_node.pre_nodes = auxiliary_pre_nodes
            auxiliary_node.next_nodes = auxiliary_next_nodes
            auxiliary_order_to_node.update({order: auxiliary_node})
            # print("auxiliary num {}, name {}, type {}, op_type {}, auxiliary {}, pre_nodes {}, next_nodes {}"
                # .format(order, auxiliary_node.get_name(), auxiliary_node._type, auxiliary_node._op_type, 
                # auxiliary_node._auxiliary, list(auxiliary_node.pre_nodes.keys()), list(auxiliary_node.next_nodes.keys())))
            
            subgraph_node = copy.deepcopy(auxiliary_node)
            subgraph_node.pre_nodes = {}
            subgraph_node.next_nodes = {}
            subgraph_node_dict.update({name: subgraph_node})
            subgraph_order_to_node.update({node.serial_number: subgraph_node})
            if name in auxiliary_start_node_name or name in original_start_node_name:
                start_node.append(subgraph_node)
            if name in auxiliary_end_node_name:
                end_node.append(subgraph_node)
        
        subgraph = LegoDNNGraph()
        subgraph.node_dict = subgraph_node_dict
        subgraph.order_to_node = subgraph_order_to_node
        subgraph.start_node = start_node
        subgraph.end_node = end_node
        
        # 由开始节点开始遍历子图 连线
        node_queue = queue.Queue()
        node_visited = set()
        for subgraph_node in start_node:
            order = subgraph_node.serial_number
            node = auxiliary_order_to_node[order]
            # print("add queue num {}, name {}, type {}, op_type {}, auxiliary {}, pre_nodes {}, next_nodes {}"
                # .format(order, node.get_name(), node._type, node._op_type, node._auxiliary, list(node.pre_nodes.keys()), list(node.next_nodes.keys())))
            node_queue.put(node)
            node_visited.add(node)
        while not node_queue.empty():
            curr_node = node_queue.get()  # 取队首节点
            curr_subgraph_node = subgraph_node_dict[curr_node.get_name()]
            for name, _ in curr_node.next_nodes.items():
                subgraph_node = subgraph_node_dict[name]
                order = subgraph_node.serial_number
                node = auxiliary_order_to_node[order]
                if node not in node_visited:
                    node_queue.put(node)
                subgraph._add_edge(curr_subgraph_node, subgraph_node)
                node_visited.add(node)
        return subgraph

    # 使用networkx可视化
    def show_graph(self, path='network.jpg'):
        G = nx.DiGraph()  # 创建空的简单有向图
        node_queue = queue.Queue()
        node_visited = set()
        pos = {}
        # 开始节点入队
        for node in self.start_node:
            node_queue.put(node)
            node_visited.add(node)
        while not node_queue.empty():
            curr_node = node_queue.get()  # 取队首节点
            for name, node in curr_node.next_nodes.items():
                if node not in node_visited:
                    node_queue.put(node)
                G.add_edge(curr_node.get_name()+' id:'+str(curr_node.serial_number), node.get_name()+' id:'+str(node.serial_number))
                node_visited.add(node)
            pos.update({curr_node.get_name()+' id:'+str(curr_node.serial_number): (curr_node.serial_number, random.randint(0,32))})
        # figure config
        plt.figure(3 ,figsize=(32,4)) 
        nx.draw(G, pos=pos, node_size=50, arrows=True, with_labels=True, font_size=5, width=0.5)
        plt.show()
        plt.savefig(path)

    def find_all_next_target_layers(self, name):
        pass
        
def topology_extraction(net, input_size, device='cuda', mode='unpack'):
    if isinstance(input_size[0], tuple):
        data = ()
        for tensor_size in input_size:
            data = data + (torch.ones(tensor_size).to(device), )
        
        data = (data, )
    else:
        data = torch.ones(input_size).to(device)
    # 通过nni建图
    if mode == 'pack':
        module_graph = build_module_graph(net, data)
    elif mode == 'unpack':
        module_graph = build_module_graph_with_unpack_manually(net, data)
    else:
        raise NotImplementedError
    # 通过nni建图
    # module_graph = build_module_graph(net, input_size)

    name_to_node = module_graph.name_to_node  # dict
    input_to_node = module_graph.input_to_node  # defaultdict
    output_to_node = module_graph.output_to_node  # dict

    # for name, node in name_to_node.items():
    #     print(node)

    graph = LegoDNNGraph()
    if mode == 'pack':
        graph.build_graph(name_to_node, input_to_node, output_to_node)
    elif mode == 'unpack':
        graph.build_graph_with_unpack_manually(name_to_node, input_to_node, output_to_node)
    
    # graph.print_ordered_node()
    # graph.show_graph()
    return graph

# def topology_extraction_data(net, input_size, device='cuda'):
#     if isinstance(input_size[0], tuple):
#         data = ()
#         for tensor_size in input_size:
#             data = data + (torch.ones(tensor_size).to(device), )
#     else:
#         data = torch.ones(input_size).to(device)
#     # 通过nni建图
#     module_graph = build_module_graph(net, data)
#     # 通过nni建图
#     # module_graph = build_module_graph(net, input_size)

#     name_to_node = module_graph.name_to_node  # dict
#     input_to_node = module_graph.input_to_node  # defaultdict
#     output_to_node = module_graph.output_to_node  # dict

#     # for name, node in name_to_node.items():
#     #     print(node)

#     graph = LegoDNNGraph()
#     graph.build_graph(name_to_node, input_to_node, output_to_node)
#     # graph.print_ordered_node()
#     # graph.show_graph()
#     return graph



if __name__=='__main__':
    from mmdet.apis import init_detector
    from functools import partial
    config = '/data/gxy/legodnn-public-version_object_detection/cv_task/object_detection/mmdet_models/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # checkpoint = '/data/gxy/legodnn-public-version_object_detection/cv_task/object_detection/mmdet_models/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    image_path = '/data/gxy/legodnn-public-version_object_detection/cv_task/object_detection/test.jpg'

    device = 'cuda'

    detector = init_detector(config, device=device) 
    detector.forward = partial(detector.legodnn_jit_forward)
    detector.eval()

    graph = topology_extraction(detector, (1,3,300,400), device=device)
    graph.print_ordered_node()

    print('------------------------------backbone-----------------------------------')
    subgraph = graph.get_subgraph('backbone')
    subgraph.print_ordered_node()

    print('Start Nodes: ')
    for node in subgraph.start_node:
        name = node.get_name()
        print(name)
    print('End Nodes: ')
    for node in subgraph.end_node:
        name = node.get_name()
        print(name)


    print('------------------------------neck-----------------------------------')
    subgraph = graph.get_subgraph('neck')
    subgraph.print_ordered_node()

    print('Start Nodes: ')
    for node in subgraph.start_node:
        name = node.get_name()
        print(name)
    print('End Nodes: ')
    for node in subgraph.end_node:
        name = node.get_name()
        print(name)