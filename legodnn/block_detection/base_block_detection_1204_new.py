import sys
import math
sys.path.insert(0, '../../')
from legodnn.block_detection.model_topology_extraction import LegoDNNGraph, LegoDNNNode
from queue import Queue
from typing import Dict, List
import copy

NOCHANGE_OUTPUR_SHAPE_LAYERS = ['Conv2d', 'Linear', 'ConvTranspose2d']
COMPRESSED_LAYERS = ['Conv2d', 'ConvTranspose2d'] # 这些是可以被压缩的层
PARAM_REUSE_LAYERS = ['Conv2d', 'Linear', 'ConvTranspose2d', 'BatchNorm2d'] # 带参数的网络层如果被重用，必须在同一个块内
BLOCK_DETECTION_MODE = ['oto']

def list_de_duplication(l: list):
    new_l = []
    for x in l:
        if x not in new_l:
            new_l.append(x)
    return new_l

class BaseBlockDetection:
    def __init__(self, graph: LegoDNNGraph, max_ratio=0.25) -> None:
        # self.mode = mode # 模式表示一个块的子图定义方式
        # oto：one to one，表示块对应的子图应该仅有一个开始节点和一个终止节点
        # otm: one to many, 表示块对应的子图仅有一个开始节点，但是可以有多个终止节点
        # mto：many to one，表示块对应的子图可以有多个输入节点，但是仅有一个终止节点
        # mtm：mant to many，表示块对应的子图有多个输入节点和多个输出节点
        
        self.graph = graph
        
        self.name_reuse_dict: Dict[List] = {} # 按名字索引所有重用层
        self.order_reuse_dict: Dict[List] = {} # 按编号索引所有重用层
        self._detection_all_reuse_layers() # 检测所有重用层
        
        # 子图约束3的条件
        self.max_ratio = max_ratio  # 一个块中最少的可压缩层
        self.min_compress_num = 2
        total_compress_num = self.get_all_compress_layer_number(self.graph)
        self.max_compress_num = max(math.floor(total_compress_num*max_ratio), 2)
        print("模型中卷积层/反卷积层总数: {}, 块中的最大比例: {}, 块中卷积层/反卷积层最小值: {}, 块中卷积层/反卷积层最大值: {}".format(total_compress_num, self.max_ratio ,self.min_compress_num, self.max_compress_num))
        self._blocks = [] # list[list]结构，或者说list[block]，block = list[int]，每个块是块中包含的module和func在graph中的serial_number
        self._blocks_no_compressed_layers = [] # list[list]结构，对应于self.blocks，存储的是每个block中的非压缩层的name
        # 记录一个块中的开始节点是否为占位符，占位符分为两种情况： 
        # 一是该块的开始节点是其它块的终止节点，那么该块的开始节点中的操作就不再进行，如果为0则不是，如果为1则是占位符
        # 二是该块的开始节点是func操作，如cat, add等,则记为占位符
        self._blocks_start_node_is_placeholder: List[List[int]] = [] 
        
        self._blocks_start_node_order: List[List[int]] = [] # 记录每个块开始节点的编号
        self._blocks_start_node_name_hook: List[List] = [] # 记录每个块的输入应该钩原始模型中层的名字，每个按照编号顺序排列
        self._blocks_start_node_hook_input_or_ouput: List[List] = []  # 记录钩输入还是输出，如果为0则钩输入，如果为1则钩输出
        self._blocks_start_node_hook_index: List[List] = [] # 记录钩出的结果在hook的输入输出list的位置
        
        self._blocks_end_node_order: List[List[int]] = [] # 记录每个块终止节点的编号
        self._blocks_end_node_name_hook: List[List] = []  # 记录每个块的输出应该钩原始模型中层的名字
        self._blocks_end_node_hook_input_or_ouput: List[List] = []  # 记录钩输入还是输出，如果为0则钩输入，如果为1则钩输出
        self._blocks_end_node_hook_index: List[List] = [] # 记录钩出的结果在hook的输入输出list的位置
        pass
    
    @property
    def blocks(self):
        return self._blocks
    @property
    def blocks_no_compressed_layers(self):
        return self._blocks_no_compressed_layers
    @property
    def blocks_start_node_is_placeholder(self):
        return self._blocks_start_node_is_placeholder
    @property
    def blocks_start_node_order(self):
        return self._blocks_start_node_order # 记录每个块开始节点的编号
    @property
    def blocks_start_node_name_hook(self):
        return self._blocks_start_node_name_hook# 记录每个块的输入应该钩原始模型中层的名字，每个按照编号顺序排列
    @property
    def blocks_start_node_hook_input_or_ouput(self):
        return self._blocks_start_node_hook_input_or_ouput  # 记录钩输入还是输出，如果为0则钩输入，如果为1则钩输出
    @property
    def blocks_start_node_hook_index(self):
        return self._blocks_start_node_hook_index # 记录钩出的结果在hook的输入输出list的位置
    @property
    def blocks_end_node_order(self):
        return self._blocks_end_node_order # 记录每个块终止节点的编号
    @property
    def blocks_end_node_name_hook(self):
        return self._blocks_end_node_name_hook  # 记录每个块的输出应该钩原始模型中层的名字
    @property
    def blocks_end_node_hook_input_or_ouput(self):
        return self._blocks_end_node_hook_input_or_ouput  # 记录钩输入还是输出，如果为0则钩输入，如果为1则钩输出
    @property
    def blocks_end_node_hook_index(self):
        return self._blocks_end_node_hook_index # 记录钩出的结果在hook的输入输出list的位置
    
    def _detection_all_reuse_layers(self): # 先找出模型中所有被重用层的编号，并且在模型中的最初索引名
        name_list = self.graph.node_dict.keys()
        reuse_dict: Dict[List] = {}
        for name in name_list:
            if self.graph.node_dict[name].get_type() !='module': # 只查看重用的module元素
                continue
            origin_name = '.'.join(name.split('.')[:-1])
            if origin_name in name_list:# 如果该层属于重用层则截取最后一个'.'之前的名字应该存在于整个图中
                if origin_name in reuse_dict: # 如果重用层已经发现，则直接加入字典
                    reuse_dict[origin_name].append(name)
                else: # 否则，将重用层与当前层一起加入字典
                    reuse_dict[origin_name] = [origin_name, name]

        self.name_reuse_dict = reuse_dict # 按名字索引

        reuse_order_dict: Dict[List] = {}
        for origin_name, reuse_list in self.name_reuse_dict.items():
            origin_order = self.graph.node_dict[origin_name].serial_number
            reuse_order_dict[origin_order] = []
            for reuse_name in reuse_list:
                reuse_order_dict[origin_order].append(self.graph.node_dict[reuse_name].serial_number)

        self.order_reuse_dict = reuse_order_dict # 按编号索引

    def get_all_compress_layer_number(self, graph: LegoDNNGraph):
        compress_num = 0
        for num, node in graph.order_to_node.items():
            assert node.get_type() in ['func', 'module'] # 目前先保证是这两个，如果遇到其它的操作类型再处理
            if node.get_op_type() in COMPRESSED_LAYERS and self._adjust_is_no_rename_node(node.get_name()):
                print("num {}, op_type {}, compress name: {}".format(num, node.get_op_type(), node.get_name()))
                compress_num = compress_num + 1
        return compress_num
    
    def get_block_all_compress_layer_number(self, block: List[int]):
        compress_num = 0
        for node_order in block:
            node = self.graph.order_to_node[node_order]
            assert node.get_type() in ['func', 'module'] # 目前先保证是这两个，如果遇到其它的操作类型再处理
            if node.get_op_type() in COMPRESSED_LAYERS and self._adjust_is_no_rename_node(node.get_name()):
                # print("num {}, op_type {}, compress name: {}".format(num, node.get_op_type(), node.get_name()))
                compress_num = compress_num + 1
        return compress_num
    
    def _adjust_is_no_rename_node(self, node_name):
        if '.'.join(node_name.split('.')[:-1]) in self.name_reuse_dict:
            return False
        else:
            return True

    def _find_module_node_in_model_name(self, node_order_or_name): # 发现一个module节点在原始模型中的名字，主要是为了处理被重用的层
        if node_order_or_name not in self.graph.order_to_node and node_order_or_name not in self.graph.node_dict:
            return None
        # 输入可以为节点的编号或者节点的名字
        assert node_order_or_name in self.graph.order_to_node or node_order_or_name in self.graph.node_dict

        if node_order_or_name in self.graph.order_to_node:
            node_name = self.graph.order_to_node[node_order_or_name].get_name()
        else:
            node_name = node_order_or_name

        if '.'.join(node_name.split('.')[:-1]) in self.name_reuse_dict:
                return '.'.join(node_name.split('.')[:-1])
        else:
            return node_name
    
    def _find_start_node_and_end_node(self, block:List[int]):
        start_node_list = []
        end_node_list = []
        for node_order in block:
            node = self.graph.order_to_node[node_order]
            
            # 判断是否为该子图输入节点，输入节点有两种
            # 1、该节点的输入为整个块的输入
            # 2、该节点的输出为整个块的输入
            if len(node.pre_nodes.items())==0:
                start_node_list.append(node_order)
            elif node in self.graph.start_node:
                start_node_list.append(node_order)
            else:
                for name, pre_node in node.pre_nodes.items():
                    serial_number = pre_node.serial_number
                    if serial_number not in block:
                        start_node_list.append(node_order)
                        break

            # 判断是否为输出节点
            # 输出节点只有一种情况，该节点在块内，且该节点的输出的整个块的输出
            if len(node.next_nodes.items())==0:
                end_node_list.append(node_order)
            else:
                for name, next_node in node.next_nodes.items():
                    serial_number = next_node.serial_number
                    if serial_number not in block:
                        end_node_list.append(node_order)
                        break
                    
        start_node_list = list_de_duplication(start_node_list)
        
        # 如果开始节点只有一个，那么该块一定只有一个输入
        # 如果开始节点大于1个，存在一种情况，该块所有的输入节点的前序节点是同一个而且不是list或者元组，那么我们将其当做该块的开始节点，但是其为占位符，仅表示该块的输入来自于它
        if len(start_node_list)>1:
            start_node_list_pre_nodes = []
            for start_node_order in start_node_list:
                start_node = self.graph.order_to_node[start_node_order]
                if len(start_node.pre_nodes.items())==0:
                    start_node_list_pre_nodes.append(start_node_order)
                else:
                    for name, pre_node in start_node.pre_nodes.items():
                        serial_number = pre_node.serial_number
                        assert serial_number!=-1
                        start_node_list_pre_nodes.append(serial_number)
                        # if serial_number!=-1: # 如果不等于-1，则其是一个前序节点
                        #     start_node_list_pre_nodes.append(serial_number)
                        # else: # 如果为-1，那么没有前序节点，该节点就是开始节点
                        #     start_node_list_pre_nodes.append(start_node_order)
            # print("开始节点的所有前序节点 {}".format())
            start_node_list = list_de_duplication(start_node_list_pre_nodes)
            
        # print("开始节点列表: {}，类型 {}, 终止节点列表: {}, 类型 {}".format(start_node_list, type(start_node_list), end_node_list, type(end_node_list)))
        return list_de_duplication(start_node_list), list_de_duplication(end_node_list)

    # 发现一个块中从输出节点开始，包含在所有路径上的第一个可压缩层
    def _find_block_all_paths_first_compressed_layer_name_from_end_nodes(self, block: List[int], end_node_list):
        assert len(end_node_list)==1
        
        def _find_block_all_paths_first_compressed_layer_name_from_output(block: List[int], end_node_order):
            end_node = self.graph.order_to_node[end_node_order]
            node_queue = Queue()
            compressed_layers_name = []
            node_queue.put(end_node)
            while not node_queue.empty():
                node = node_queue.get()
                if node.serial_number not in block:
                    continue
                # if node.get_op_type() in ONCHANGE_OUTPUR_SHAPE_LAYERS:
                if node.get_op_type() in COMPRESSED_LAYERS:
                    compressed_layers_name.append(self._find_module_node_in_model_name(node.get_name()))
                    continue
                else:
                    for pre_name, pre_node in node.pre_nodes.items():
                        node_queue.put(pre_node)
            return list_de_duplication(compressed_layers_name)

        block_end_compressed_layers = []
        for end_node_order in end_node_list:
            compressed_layers_name = _find_block_all_paths_first_compressed_layer_name_from_output(block, end_node_order)
            block_end_compressed_layers = block_end_compressed_layers + compressed_layers_name

        return list_de_duplication(block_end_compressed_layers)

    def _adjust_all_param_reuse_layers_in_block(self, block: List[int]):
        for node_order in block:
            if self.graph.order_to_node[node_order].get_op_type() not in PARAM_REUSE_LAYERS:
                continue
            
            node_name = self.graph.order_to_node[node_order].get_name()
            reuse_layer_name = None 
            if node_name in self.name_reuse_dict: # 如果当前层是重用层, 且名字是重用层在原始模型中的名字
                reuse_layer_name = node_name

            if '.'.join(node_name.split('.')[:-1]) in self.name_reuse_dict: # 如果当前层是重用层，且重用层是nni按照编号重起的名字
                reuse_layer_name = '.'.join(node_name.split('.')[:-1])
            
            if reuse_layer_name is not None: # 该层是重用层
                reuse_layer_order = self.graph.node_dict[reuse_layer_name].serial_number
                for order_reuse in self.order_reuse_dict[reuse_layer_order]: # 判断当前重用是否都在块内
                    if order_reuse not in block: # 存在重用层不在块内，则不满足块条件
                        return False
        return True
    
    def _find_block_hook_nodes_input_and_output(self,block: List[int], start_node_list: List[int], end_node_list: List[int]):
        assert len(start_node_list)==1
        assert len(end_node_list)==1
        
        def _find_hook_index(node_name: str): # 发现重用层和非重用层的勾出索引，如果为非重用层，因为hook数组中只有一个元素，则勾出索引为0，如果为重用层，则判断应该勾出元素在hook数组中的位置
            if '.'.join(node_name.split('.')[:-1]) in self.name_reuse_dict:
                return int(node_name.split('.')[-1])
            else:
                return int(0)
                
        def _find_block_start_nodes_hook_input_and_output(block: List[int], start_node_order):
            # 判断当前划分出的块的输入能否被勾出，且找到需要勾出的块
            # update 2021.11.3 16:46: 增加可重用层勾出的输入输出在hook-list位置 
            # update 2021.11.13 15:16: 
            start_node_hook = -1
            start_node_hook_input_or_ouput = -1
            start_node_hook_index = 0
            
            # print(start_node_is_placeholder)
            # 查找如何勾出块的输入
            if start_node_order not in block: # 如果开始节点不在块内
                # print("开始节点的名字: {}, 图中的所有节点: {}".format(start_node_order, self.graph.order_to_node.keys()))
                if self.graph.order_to_node[start_node_order].get_type() == 'module':  #如果开始节点为module，钩输出
                    start_node_hook = start_node_order
                    start_node_hook_input_or_ouput = 1
                else:
                    next_nodes = self.graph.order_to_node[start_node_order].next_nodes #如果开始节点为其它，则钩后续节点的输出，且这个后续节点必须在块内
                    for name, next_node in next_nodes.items():
                        if next_node.get_type()=='module' and next_node.serial_number in block:
                            start_node_hook = next_node.serial_number
                            start_node_hook_input_or_ouput = 0
            else: # 如果开始节点在块内
                if self.graph.order_to_node[start_node_order].get_type() == 'module': #如果开始节点为module，钩输入
                    start_node_hook = start_node_order
                    start_node_hook_input_or_ouput = 0
                else:
                    pre_nodes = self.graph.order_to_node[start_node_order].pre_nodes #如果开始节点为其它，则钩前续节点的输出
                    for name, pre_node in pre_nodes.items():
                        if pre_node.get_type()=='module':
                            start_node_hook = pre_node.serial_number
                            start_node_hook_input_or_ouput = 1

            if start_node_hook==-1:
                start_node_name_hook = ''
                start_node_hook_index = -1
            else:
                start_node_name_hook = self._find_module_node_in_model_name(self.graph.order_to_node[start_node_hook].get_name())
                start_node_hook_index = _find_hook_index(self.graph.order_to_node[start_node_hook].get_name())

            out = (start_node_name_hook, start_node_hook_input_or_ouput, start_node_hook_index)
            return out

        # 判断当前划分出的块的输入输出能否被勾出，且找到需要勾出的块
        # update 2021.11.3 16:46: 增加可重用层勾出的输入输出在hook-list位置

        def _find_block_end_nodes_hook_input_and_output(block: List[int], end_node_order):
            assert end_node_order in block
            # 判断当前划分出的块的输出能否被勾出，且找到需要勾出的块
            # update 2021.11.3 16:46: 增加可重用层勾出的输入输出在hook-list位置 
    
            end_node_hook = -1
            end_node_hook_input_or_output = -1
            end_node_hook_index = 0

            # 查找如何勾出块的输出
            if self.graph.order_to_node[end_node_order].get_type() == 'module':
                end_node_hook = end_node_order
                end_node_hook_input_or_output = 1
            else:
                next_nodes = self.graph.order_to_node[end_node_order].next_nodes
                for name, next_node in next_nodes.items():
                    if next_node.get_type()=='module':
                        end_node_hook = next_node.serial_number
                        end_node_hook_input_or_output = 0
            
            if end_node_hook == -1:
                end_node_name_hook = ''
                end_node_hook_index = -1
            else:
                end_node_name_hook = self._find_module_node_in_model_name(self.graph.order_to_node[end_node_hook].get_name())
                end_node_hook_index = _find_hook_index(self.graph.order_to_node[end_node_hook].get_name())

            out = (end_node_name_hook, end_node_hook_input_or_output, end_node_hook_index)
            return out

        start_nodes_name_hook = []
        start_nodes_hook_input_or_ouput = []
        start_nodes_hook_index = []
        
        end_nodes_name_hook = []
        end_nodes_hook_input_or_output = []
        end_nodes_hook_index = []
        
        # assert len(start_nodes_is_placeholder) == len(start_node_list)
        for start_node_order in start_node_list:
            out = _find_block_start_nodes_hook_input_and_output(block, start_node_order)
            start_nodes_name_hook.append(out[0])
            start_nodes_hook_input_or_ouput.append(out[1])
            start_nodes_hook_index.append(out[2])
        
        for end_node_order in end_node_list:
            out = _find_block_end_nodes_hook_input_and_output(block, end_node_order)
            end_nodes_name_hook.append(out[0])
            end_nodes_hook_input_or_output.append(out[1])
            end_nodes_hook_index.append(out[2])
        
        hook_tuple = (start_nodes_name_hook, start_nodes_hook_input_or_ouput, start_nodes_hook_index, end_nodes_name_hook, end_nodes_hook_input_or_output, end_nodes_hook_index)
        
        for hook_list in hook_tuple:
            for i in hook_list:
                if i==-1:
                    return None
        
        return hook_tuple
    
    def _get_block_input_num(self, start_node_list: List[int], block: List):
        block_input_num = 0
        for start_node_order in start_node_list:
            if start_node_order not in block:
                block_input_num +=1
            elif len(self.graph.order_to_node[start_node_order].pre_nodes.items())==0:
                block_input_num +=1
            else:
                block_input_num += len(self.graph.order_to_node[start_node_order].pre_nodes.items())
        
        return block_input_num
        
    def detection_all_block(self):
        for num, node in self.graph.order_to_node.items():
            assert node.get_type() in ['func', 'module'] # 目前先保证是这两个，如果遇到其它的操作类型再处理
        
        blocks = [] # list[list]结构，或者说list[block]，block = list[int]，每个块是块中包含的module和func在graph中的serial_number
        blocks_no_compressed_layers = [] # list[list]结构，对应于self.blocks，存储的是每个block中的非压缩层的name

        # 记录一个块中的开始节点是否为占位符，占位符分为两种情况： 
        # 一是该块的开始节点是其它块的终止节点，那么该块的开始节点中的操作就不再进行，如果为0则不是，如果为1则是占位符
        # 二是该块的开始节点是func操作，如cat, add等,则记为占位符
        blocks_start_node_is_placeholder: List[List[int]] = [] 
        
        blocks_start_node_order: List[List[int]] = [] # 记录每个块开始节点的编号
        blocks_start_node_name_hook: List[List] = [] # 记录每个块的输入应该钩原始模型中层的名字，每个按照编号顺序排列
        blocks_start_node_hook_input_or_ouput: List[List] = []  # 记录钩输入还是输出，如果为0则钩输入，如果为1则钩输出
        blocks_start_node_hook_index: List[List] = [] # 记录钩出的结果在hook的输入输出list的位置
        
        blocks_end_node_order: List[List[int]] = [] # 记录每个块终止节点的编号
        blocks_end_node_name_hook: List[List] = []  # 记录每个块的输出应该钩原始模型中层的名字
        blocks_end_node_hook_input_or_ouput: List[List] = []  # 记录钩输入还是输出，如果为0则钩输入，如果为1则钩输出
        blocks_end_node_hook_index: List[List] = [] # 记录钩出的结果在hook的输入输出list的位置
        
        # node_sum = len(self.graph.order_to_node)
        node_max = max(self.graph.order_to_node.keys())
        start_order = min(self.graph.order_to_node.keys())
                        
        while(start_order < node_max):
            block_no_compressed_layers_tmp = [] # list[list]结构，对应于self.blocks，存储的是每个block中的非压缩层的name
            block_start_node_is_placeholder_tmp: List[int] = [] 
            block_start_node_order_tmp: List[int] = [] # 记录每个块开始节点的编号
            block_start_node_name_hook_tmp: List = [] # 记录每个块的输入应该钩原始模型中层的名字，每个按照编号顺序排列
            block_start_node_hook_input_or_ouput_tmp: List = []  # 记录钩输入还是输出，如果为0则钩输入，如果为1则钩输出
            block_start_node_hook_index_tmp: List = [] # 记录钩出的结果在hook的输入输出list的位置
            
            block_end_node_order_tmp: List[int] = [] # 记录每个块终止节点的编号
            block_end_node_name_hook_tmp: List = []  # 记录每个块的输出应该钩原始模型中层的名字
            block_end_node_hook_input_or_ouput_tmp: List = []  # 记录钩输入还是输出，如果为0则钩输入，如果为1则钩输出
            block_end_node_hook_index_tmp: List = [] # 记录钩出的结果在hook的输入输出list的位置
        
            compress_num = 0
            compress_num_tmp = 0
            # node_order = start_order
            block = []
            # while node_order<=node_sum:
            block_tmp = copy.deepcopy(block)
            
            for node_order in range(start_order, node_max+1):
                # print(node_order)
                block.append(node_order)
 
                if self.graph.order_to_node[node_order].get_op_type() in COMPRESSED_LAYERS and self._adjust_is_no_rename_node(self.graph.order_to_node[node_order].get_name()):
                    compress_num += 1
                    
                if compress_num < self.min_compress_num:
                    continue
                
                elif compress_num<=self.max_compress_num:
                    # 计算当前块的输入和输出总数x,y
                    # print("block: {}".format(block))
                    start_node_list, end_node_list = self._find_start_node_and_end_node(block)
                    
                    # 计算一个块的输入数量
                    block_input_num = self._get_block_input_num(start_node_list, block)
                    block_output_num = len(end_node_list)
                    # print("开始节点列表: {}，终止节点列表: {}， 块的输入个数: {}，块的输出个数: {}".format(start_node_list, end_node_list, block_input_num, block_output_num))
                    if block_input_num>1 or block_output_num>1:
                        continue
                    
                    # 计算当前块中可以被压缩的卷积/反卷积层总数z
                    block_no_compressed_layers = self._find_block_all_paths_first_compressed_layer_name_from_end_nodes(block, end_node_list)
                    if (compress_num-len([layer in COMPRESSED_LAYERS for layer in block_no_compressed_layers]))<=0:
                        continue
                    
                    # 带参数的所有重用层必须要在同一个块内
                    reuse_layers_in_block = self._adjust_all_param_reuse_layers_in_block(block)
                    if not reuse_layers_in_block:
                        continue
                    
                    # print("开始节点: {}, 块的输入数量: {}, 块的输出数量: {}".format(start_node_list, block_input_num, block_output_num))
                    # 判断块的输入输出能否被勾出以及勾出的位置
                    hook_tuple = self._find_block_hook_nodes_input_and_output(block, start_node_list, end_node_list) # 
                        
                    if hook_tuple is not None and compress_num > compress_num_tmp:
                        
                        block_no_compressed_layers_tmp = copy.deepcopy(block_no_compressed_layers)
                        block_start_node_is_placeholder_tmp = [start_node_order not in block for start_node_order in start_node_list]
                        block_tmp = copy.deepcopy(block)
                        block_tmp = list_de_duplication(start_node_list + block_tmp)
                        block_start_node_order_tmp = start_node_list
                        block_start_node_name_hook_tmp = hook_tuple[0]
                        block_start_node_hook_input_or_ouput_tmp = hook_tuple[1]
                        block_start_node_hook_index_tmp = hook_tuple[2]
                        
                        block_end_node_order_tmp = end_node_list
                        block_end_node_name_hook_tmp = hook_tuple[3]
                        block_end_node_hook_input_or_ouput_tmp = hook_tuple[4]
                        block_end_node_hook_index_tmp = hook_tuple[5]
                        
                        compress_num_tmp = compress_num
                        start_order = node_order + 1
                else:
                    break
                
            if len(block_tmp)>0:
                blocks.append(block_tmp)
                blocks_no_compressed_layers.append(block_no_compressed_layers_tmp)

                blocks_start_node_is_placeholder.append(block_start_node_is_placeholder_tmp)
                blocks_start_node_order.append(block_start_node_order_tmp)
                blocks_start_node_name_hook.append(block_start_node_name_hook_tmp)
                blocks_start_node_hook_input_or_ouput.append(block_start_node_hook_input_or_ouput_tmp)
                blocks_start_node_hook_index.append(block_start_node_hook_index_tmp)
                
                blocks_end_node_order.append(block_end_node_order_tmp)
                blocks_end_node_name_hook.append(block_end_node_name_hook_tmp)
                blocks_end_node_hook_input_or_ouput.append(block_end_node_hook_input_or_ouput_tmp)
                blocks_end_node_hook_index.append(block_end_node_hook_index_tmp)
            else:
                start_order = start_order + 1

                        
        self._blocks = blocks # list[list]结构，或者说list[block]，block = list[int]，每个块是块中包含的module和func在graph中的serial_number
        self._blocks_no_compressed_layers = blocks_no_compressed_layers # list[list]结构，对应于self.blocks，存储的是每个block中的非压缩层的name

        # 记录一个块中的开始节点是否为占位符，占位符分为两种情况： 
        # 一是该块的开始节点是其它块的终止节点，那么该块的开始节点中的操作就不再进行，如果为0则不是，如果为1则是占位符
        # 二是该块的开始节点是func操作，如cat, add等,则记为占位符
        self._blocks_start_node_is_placeholder = blocks_start_node_is_placeholder
        
        self._blocks_start_node_order = blocks_start_node_order # 记录每个块开始节点的编号
        self._blocks_start_node_name_hook = blocks_start_node_name_hook # 记录每个块的输入应该钩原始模型中层的名字，每个按照编号顺序排列
        self._blocks_start_node_hook_input_or_ouput = blocks_start_node_hook_input_or_ouput  # 记录钩输入还是输出，如果为0则钩输入，如果为1则钩输出
        self._blocks_start_node_hook_index = blocks_start_node_hook_index # 记录钩出的结果在hook的输入输出list的位置
        
        self._blocks_end_node_order = blocks_end_node_order # 记录每个块终止节点的编号
        self._blocks_end_node_name_hook = blocks_end_node_name_hook # 记录每个块的输出应该钩原始模型中层的名字
        self._blocks_end_node_hook_input_or_ouput = blocks_end_node_hook_input_or_ouput  # 记录钩输入还是输出，如果为0则钩输入，如果为1则钩输出
        self._blocks_end_node_hook_index = blocks_end_node_hook_index # 记录钩出的结果在hook的输入输出list的位置

    def print_blocks(self):
        assert len(self._blocks) == len(self._blocks_start_node_name_hook) == len(self._blocks_start_node_hook_input_or_ouput) == len(self._blocks_end_node_name_hook) == len(self._blocks_end_node_hook_input_or_ouput)

        num = 1
        for i, block in enumerate(self._blocks):
            assert len(self._blocks_start_node_is_placeholder[i]) == len(self._blocks_start_node_order[i]) == len(self._blocks_start_node_name_hook[i]) == len(self._blocks_start_node_hook_input_or_ouput[i]) == len(self._blocks_start_node_hook_index[i])

            assert len(self._blocks_end_node_order[i]) == len(self._blocks_end_node_name_hook[i]) == len(self._blocks_end_node_hook_input_or_ouput[i]) == len(self._blocks_end_node_hook_index[i])

            print("第{}个块, 有{}个输入节点，有{}个输出节点, 有{}个卷积层/反卷积层".format(num, len(self._blocks_start_node_name_hook[i]), len(self._blocks_end_node_name_hook[i]), self.get_block_all_compress_layer_number(block)))
            
            for index, (start_node_is_placeholder, start_node_order, start_node_name, start_node_hook_input_or_output, start_node_hook_index) in enumerate(zip(self._blocks_start_node_is_placeholder[i], self._blocks_start_node_order[i], self._blocks_start_node_name_hook[i], self._blocks_start_node_hook_input_or_ouput[i], self._blocks_start_node_hook_index[i])):

                if start_node_is_placeholder:
                    print("第{}个开始节点是占位符, 索引为{}, 名字为{}".format(index+1, start_node_order, self.graph.order_to_node[start_node_order].get_name()), end="  ---->  ")
                else:
                    print("第{}个开始节点不是占位符, 索引为{}, 名字为{}".format(index+1, start_node_order, self.graph.order_to_node[start_node_order].get_name()), end="  ---->  ")

                if start_node_hook_input_or_output==0:
                    print("输入：{}的输入, 索引为{}".format(start_node_name, start_node_hook_index))
                elif start_node_hook_input_or_output==1:
                    print("输入：{}的输出, 索引为{}".format(start_node_name, start_node_hook_index))
                else:
                    raise NotImplementedError

            for index, (end_node_order, end_node_name, end_node_hook_input_or_output, end_node_hook_index) in enumerate(zip(self._blocks_end_node_order[i], self._blocks_end_node_name_hook[i], self._blocks_end_node_hook_input_or_ouput[i], self._blocks_end_node_hook_index[i])):

                print("第{}个终止节点, 索引为{}, 名字为{}".format(index+1, end_node_order, self.graph.order_to_node[end_node_order].get_name()), end="  ---->   ")

                if end_node_hook_input_or_output==0:
                    print("输出：{}的输入, 索引为{}".format(end_node_name, end_node_hook_index))
                elif end_node_hook_input_or_output==1:
                    print("输出：{}的输出, 索引为{}".format(end_node_name, end_node_hook_index))
                else:
                    raise NotImplementedError

            for order in block:
                print("层的编号：{}, 层的名字：{}".format(self.graph.order_to_node[order].serial_number, self.graph.order_to_node[order].get_name()))

            print("非可压缩层如下：")
            print(self._blocks_no_compressed_layers[i])
            num = num + 1
    
    def print_reuse_layers(self): # 打印所有重用层
        # print(1111)
        for origin_name, origin_order in zip(self.name_reuse_dict.keys(), self.order_reuse_dict.keys()):
            print("重用层 order {}, name {}".format(origin_order, origin_name))
            print(self.order_reuse_dict[origin_order])
            print(self.name_reuse_dict[origin_name])