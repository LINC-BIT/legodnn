from re import L
import re
import sys

from networkx.classes.function import non_edges
sys.path.insert(0, '../../')
from legodnn.block_detection.model_topology_extraction import LegoDNNGraph, LegoDNNNode
from queue import Queue
from typing import Dict, List
import copy

COMPRESSED_LAYERS = ['Conv2d', 'Linear', 'ConvTranspose2d'] # 这些是可以被压缩的层
PARAM_REUSE_LAYERS = COMPRESSED_LAYERS + ['BatchNorm2d'] # 带参数的网络层如果被重用，必须在同一个块内
BLOCK_DETECTION_MODE = ['oto', 'otm', 'mto', 'mtm']

class BaseBlockDetection:
    def __init__(self, graph: LegoDNNGraph, min_compress_num=2, max_compress_num=4, mode='oto') -> None:
        self.mode = mode # 模式表示一个块的子图定义方式
        # oto：one to one，表示块对应的子图应该仅有一个开始节点和一个终止节点
        # otm: one to many, 表示块对应的子图仅有一个开始节点，但是可以有多个终止节点
        # mto：many to one，表示块对应的子图可以有多个输入节点，但是仅有一个终止节点
        # mtm：mant to many，表示块对应的子图有多个输入节点和多个输出节点
        assert self.mode in BLOCK_DETECTION_MODE
        self.graph = graph
        
        # 子图约束3的条件
        self.min_compress_num = min_compress_num  # 一个块中最少的可压缩层
        self.max_compress_num = max_compress_num  # 一个块中最多的可压缩层

        self.name_reuse_dict: Dict[List] = {} # 按名字索引所有重用层
        self.order_reuse_dict: Dict[List] = {} # 按编号索引所有重用层
        self._detection_all_reuse_layers() # 检测所有重用层
        
        
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
    
    def get_all_compress_layer_number(self):
        compress_num = 0
        for num, node in self.graph.order_to_node.items():
            assert node.get_type() in ['func', 'module'] # 目前先保证是这两个，如果遇到其它的操作类型再处理
            if node.get_op_type() in COMPRESSED_LAYERS:
                compress_num = compress_num + 1
        return compress_num

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

    def _adjust_is_no_rename_node(self, node_name):
        if '.'.join(node_name.split('.')[:-1]) in self.name_reuse_dict:
            return False
        else:
            return True

    def _adjust_is_subgraph(self, block: List[int]):
        assert self.mode in BLOCK_DETECTION_MODE
        
        start_node_list, end_node_list = self._find_all_start_and_end_nodes(block)
        # print("开始节点: {}, 终止节点: {}".format(start_node_list, end_node_list))
        if self.mode in ['oto', 'otm']:
            # 有多个输入节点就不是一个块
            if len(start_node_list)>1:
                return False

        if self.mode in ['oto', 'mto']:
            # 有多输出节点，就不是一个块
            if len(end_node_list)>1:
                return False
                
        if self.mode in ['oto', 'otm']:
            # 至少有一个开始节点,且开始节点是编号最小的节点
            if len(start_node_list)!=1 or start_node_list[0]!=min(block):
                return False

        if self.mode in ['oto', 'mto']:
            # 至少有一个终止节点，终止节点是编号最大的节点
            if len(end_node_list)!=1 or end_node_list[0]!=max(block):
                return False

        if self.mode in ['mtm']:
            #保证至少有一个开始节点和一个终止节点
            if len(start_node_list)<1 or len(end_node_list)<1:
                return False
        
        return True
    
    
    def _adjust_is_block(self, block: List[int], blocks: List[List], compress_num_flag=True):
        # 发现所有的开始节点和终止节点
        start_node_list, end_node_list = self._find_all_start_and_end_nodes(block)
        
        # 判断开始节点是否为占位符，占位符的意思是该节点是上一个块的输出节点或者是func操作
        start_nodes_is_placeholder = self._adjust_start_nodes_is_placeholder(start_node_list, blocks)
        
        # v2 判断是否满足重用层条件: 每个块内都不允许存在带参数的重用层
        # 现在这种方法比较蠢
        for node_order in block:
            if node_order in start_node_list and start_nodes_is_placeholder[start_node_list.index(node_order)]==0:
                continue 
            
            if self.graph.order_to_node[node_order].get_op_type() not in PARAM_REUSE_LAYERS:
                continue
            
            node_name = self.graph.order_to_node[node_order].get_name()
            if '.'.join(node_name.split('.')[:-1]) in self.name_reuse_dict: # 如果当前层是重用层，且重用层是nni按照编号重起的名字, 那么
                return False   

         # 计算可压缩层数量， NOTE：待做：重用的层只计算一次
        if compress_num_flag:
            compress_num = 0
            for node_order in block:
                if node_order in start_node_list and start_nodes_is_placeholder[start_node_list.index(node_order)]:
                    continue 
                node = self.graph.order_to_node[node_order]
                if node.get_op_type() in COMPRESSED_LAYERS and self._adjust_is_no_rename_node(node.get_name()):
                    compress_num = compress_num + 1
            
            # 判断可压缩层是否满足成为一个块的条件
            if compress_num < self.min_compress_num or compress_num > self.max_compress_num:
                return False

        # 判断是否至少存在一条路径，该路径上至少有两个可压缩层 NOTE：这并不能保证块一个可被压缩，但是先这样简单判断一下 NOTE:这里貌似写错了，之后再研究吧
        block_start_node_list, block_end_node_list = self._find_all_start_and_end_nodes(block)
        start_input_compressed_layers = self._find_block_all_paths_first_compressed_layer_name_from_start_nodes(block, block_start_node_list)
        start_output_compressed_layers = self._find_block_all_paths_first_compressed_layer_name_from_end_nodes(block, block_end_node_list)

        assert len(start_input_compressed_layers) > 0
        assert len(start_output_compressed_layers) > 0

        # 如果从输入节点每条路径上遇到的可压缩层与从输出节点每条路径上遇到的可压缩层相同，则每条路径最多存在一个可压缩层，则该块一定不存在
        input_flag = False 
        for node_name in start_input_compressed_layers:
            if node_name not in start_output_compressed_layers:
                input_flag = True
                break

        output_flag = False
        for node_name in start_output_compressed_layers:
            if node_name not in start_input_compressed_layers:
                output_flag = True   
                break       
        if input_flag or output_flag:
            return True
        else:
            return False
        
        return True

    def _adjust_block_hook_nodes_input_and_output(self, block: List[int], start_nodes_is_placeholder, start_node_list, end_node_list):
        def _find_block_start_nodes_hook_input_and_output(block: List[int], start_node_is_placeholder: bool, start_node_order):
            # 判断当前划分出的块的输入能否被勾出，且找到需要勾出的块
            # update 2021.11.3 16:46: 增加可重用层勾出的输入输出在hook-list位置 
            # update 2021.11.13 15:16: 
            start_node_hook = -1
            start_node_hook_input_or_ouput = -1
            start_node_hook_index = 0
            
            # print(start_node_is_placeholder)
            # 查找如何勾出块的输入
            if start_node_is_placeholder: # 如果开始节点是占位符
                if self.graph.order_to_node[start_node_order].get_type() == 'module':  
                    start_node_hook = start_node_order
                    start_node_hook_input_or_ouput = 1
                else:
                    next_nodes = self.graph.order_to_node[start_node_order].next_nodes
                    for name, next_node in next_nodes.items():
                        if next_node.get_type()=='module':
                            start_node_hook = next_node.serial_number
                            start_node_hook_input_or_ouput = 0
            else:
                if self.graph.order_to_node[start_node_order].get_type() == 'module':
                    start_node_hook = start_node_order
                    start_node_hook_input_or_ouput = 0
                else:
                    pre_nodes = self.graph.order_to_node[start_node_order].pre_nodes
                    for name, pre_node in pre_nodes.items():
                        if pre_node.get_type()=='module':
                            start_node_hook = pre_node.serial_number
                            start_node_hook_input_or_ouput = 1
            
            def _find_hook_index(node_name: str): # 发现重用层和非重用层的勾出索引，如果为非重用层，因为hook数组中只有一个元素，则勾出索引为0，如果为重用层，则判断应该勾出元素在hook数组中的位置
                if '.'.join(node_name.split('.')[:-1]) in self.name_reuse_dict:
                    return int(node_name.split('.')[-1])
                else:
                    return int(0)

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
            
            def _find_hook_index(node_name: str): # 发现重用层和非重用层的勾出索引，如果为非重用层，因为hook数组中只有一个元素，则勾出索引为0，如果为重用层，则判断应该勾出元素在hook数组中的位置
                if '.'.join(node_name.split('.')[:-1]) in self.name_reuse_dict:
                    return int(node_name.split('.')[-1])
                else:
                    return int(0)

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
        assert len(start_nodes_is_placeholder) == len(start_node_list)
        for start_node_is_placeholder, start_node_order in zip(start_nodes_is_placeholder, start_node_list):
            out = _find_block_start_nodes_hook_input_and_output(block, start_node_is_placeholder, start_node_order)
            start_nodes_name_hook.append(out[0])
            start_nodes_hook_input_or_ouput.append(out[1])
            start_nodes_hook_index.append(out[2])
        
        for end_node_order in end_node_list:
            out = _find_block_end_nodes_hook_input_and_output(block, end_node_order)
            end_nodes_name_hook.append(out[0])
            end_nodes_hook_input_or_output.append(out[1])
            end_nodes_hook_index.append(out[2])
        
        return (start_nodes_name_hook, start_nodes_hook_input_or_ouput, start_nodes_hook_index, end_nodes_name_hook, end_nodes_hook_input_or_output, end_nodes_hook_index)

    def _adjust_start_nodes_is_placeholder(self, start_nodes_list, blocks: List[List]):
        def _adjust_start_node_is_placeholder(node_order):
            if self.graph.order_to_node[node_order].get_type() in ['func']: # 如果开始节点为func，则直接设置为占位符
                return True
            # 如果开始节点在其它块中，则设置为占位符
            for block in blocks:
                if node_order in block:
                    return True
            return False
        
        start_nodes_is_placeholder = []
        for start_node in start_nodes_list:
            start_nodes_is_placeholder.append(_adjust_start_node_is_placeholder(start_node))

        return start_nodes_is_placeholder

    # 发现一个块中从输入节点开始，包含在所有路径上的第一个可压缩层
    def _find_block_all_paths_first_compressed_layer_name_from_start_nodes(self, block: List[int], start_node_list):
        def _find_block_all_paths_first_compressed_layer_name_from_input(block: List[int], start_node_order):
            start_node = self.graph.order_to_node[start_node_order]
            node_queue = Queue()
            compressed_layers_name = []
            node_queue.put(start_node)
            while not node_queue.empty():
                node = node_queue.get()
                if node.serial_number not in block: 
                    continue
                if node.get_op_type() in COMPRESSED_LAYERS:
                    compressed_layers_name.append(self._find_module_node_in_model_name(node.get_name()))
                    continue
                else:
                    for next_name, next_node in node.next_nodes.items():
                        node_queue.put(next_node)
            
            return list(set(compressed_layers_name))
        
        block_start_compressed_layers = []
        for start_node_order in start_node_list:
            compressed_layers_name = _find_block_all_paths_first_compressed_layer_name_from_input(block, start_node_order)
            block_start_compressed_layers = block_start_compressed_layers + compressed_layers_name
        return list(set(block_start_compressed_layers))

    # 发现一个块中从输出节点开始，包含在所有路径上的第一个可压缩层
    def _find_block_all_paths_first_compressed_layer_name_from_end_nodes(self, block: List[int], end_node_list):
        def _find_block_all_paths_first_compressed_layer_name_from_output(block: List[int], end_node_order):
            end_node = self.graph.order_to_node[end_node_order]
            node_queue = Queue()
            compressed_layers_name = []
            node_queue.put(end_node)
            while not node_queue.empty():
                node = node_queue.get()
                if node.serial_number not in block:
                    continue
                if node.get_op_type() in COMPRESSED_LAYERS:
                    compressed_layers_name.append(self._find_module_node_in_model_name(node.get_name()))
                    continue
                else:
                    for pre_name, pre_node in node.pre_nodes.items():
                        node_queue.put(pre_node)
            return list(set(compressed_layers_name))

        block_end_compressed_layers = []
        for end_node_order in end_node_list:
            compressed_layers_name = _find_block_all_paths_first_compressed_layer_name_from_output(block, end_node_order)
            block_end_compressed_layers = block_end_compressed_layers + compressed_layers_name

        # # NOTE: ??? ，这种情况不可能出现
        # for node_order in block:
        #     if self.graph.order_to_node[node_order] in self.graph.end_node:
        #         compressed_layers_name = _find_block_all_paths_first_compressed_layer_name_from_output(block, node_order)
        #         block_end_compressed_layers = block_end_compressed_layers + compressed_layers_name

        return list(set(block_end_compressed_layers))

    def _find_all_start_and_end_nodes(self, block: List[int]):
        start_node_list = []
        end_node_list = []
        for node_order in block:
            node = self.graph.order_to_node[node_order]

            # 判断是否为输入节点
            if len(node.pre_nodes)==0:
                start_node_list.append(node_order)
            elif node in self.graph.start_node:
                start_node_list.append(node_order)
            else:
                for name, pre_node in node.pre_nodes.items():
                    serial_number = pre_node.serial_number
                    if serial_number==-1 or serial_number not in block:
                        start_node_list.append(node_order)
                        break

            # 判断是否为输出节点
            if len(node.next_nodes)==0:
                end_node_list.append(node_order)
            elif node in self.graph.end_node:
                compressed_layers = self._find_block_all_paths_first_compressed_layer_name_from_end_nodes(block, [node_order])
                if len(compressed_layers) > 0:
                    end_node_list.append(node_order)
            else:
                for name, next_node in node.next_nodes.items():
                    serial_number = next_node.serial_number
                    if serial_number==-1 or serial_number not in block:
                        end_node_list.append(node_order)
                        break
                    
        return start_node_list, end_node_list

    def detection_all_basic_block(self):
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
            block = []
            node_order = start_order
            find_block_flag = False
            block.append(node_order)
            node_order = node_order + 1
            # while node_order<=node_sum:
            while node_order<=node_max:
                block.append(node_order)
                # print(block)

                # 判断是否为一个子图, 即块的子图约束1
                is_subgraph = self._adjust_is_subgraph(block)
            
                if is_subgraph:
                    # 判断是否满足块的子图约束其它条件：1、所有的可重用层必须在同一个块内；2、满足块可压缩要求(至少存在一个子图，该子图中所有从开始节点到终止节点的路径上都至少有两个可压缩层) NOTE：目前的实现属于残缺版；3、满足块中可压缩层数要求
                    is_block = self._adjust_is_block(block, blocks)
                else:
                    is_block = False
                
                # 发现一个满足条件的子图
                if is_subgraph and is_block:
                    # exit(0)
                    # 发现所有的开始节点和终止节点
                    start_node_list, end_node_list = self._find_all_start_and_end_nodes(block)
                    
                    # 判断开始节点是否为占位符，占位符的意思是该节点是上一个块的输出节点或者是func操作
                    start_nodes_is_placeholder = self._adjust_start_nodes_is_placeholder(start_node_list, blocks)
                    
                    # # 判断当前子图的输入输出能否被勾出，能则是一个可被训练的块，不能即使满足子图约束条件也丢弃，因为无法实现块的训练。
                    # hook_tuple = self._adjust_block_hook_node_input_and_output(block, start_node_is_placeholder) # (start_node_name_hook, start_node_hook_input_or_ouput, start_node_hook_index, end_node_name_hook, end_node_hook_input_or_output, start_node_hook_index)
                    # 判断当前子图的输入输出能否被勾出，能则是一个可被训练的块，不能即使满足子图约束条件也丢弃，因为无法实现块的训练。
                    hook_tuple = self._adjust_block_hook_nodes_input_and_output(block, start_nodes_is_placeholder, start_node_list, end_node_list) # (start_node_name_hook, start_node_hook_input_or_ouput, start_node_hook_index, end_node_name_hook, end_node_hook_input_or_output, start_node_hook_index)
                    
                    assert len(hook_tuple)==6
                    block_able_train = True
                    for hook_list in hook_tuple:
                        for i in hook_list:
                            if i==-1:
                                block_able_train=False
                                break
                    
                    if block_able_train: 
                        # 当前块能被训练，且钩出当前块输入输出的节点
                        blocks_no_compressed_layers.append(self._find_block_all_paths_first_compressed_layer_name_from_end_nodes(block, end_node_list))

                        blocks_start_node_is_placeholder.append(start_nodes_is_placeholder)
                        blocks_start_node_order.append(start_node_list)
                        blocks_start_node_name_hook.append(hook_tuple[0])
                        blocks_start_node_hook_input_or_ouput.append(hook_tuple[1])
                        blocks_start_node_hook_index.append(hook_tuple[2])
                        
                        blocks_end_node_order.append(end_node_list)
                        blocks_end_node_name_hook.append(hook_tuple[3])
                        blocks_end_node_hook_input_or_ouput.append(hook_tuple[4])
                        blocks_end_node_hook_index.append(hook_tuple[5])
                        find_block_flag = True
                        break

                # 未发现满足条件的子图且还有节点未遍历
                node_order = node_order + 1
            
            if find_block_flag: # 找到一个块，从该块的结束节点开始遍历
                blocks.append(block)
                start_order = node_order
            else:
                start_order = start_order + 1 # 未找到块，则从当前节点的下一个节点开始遍历

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
        
        # return blocks, blocks_no_compressed_layers, blocks_start_node_is_placeholder, blocks_start_node_order, blocks_start_node_name_hook, blocks_start_node_hook_input_or_ouput, blocks_start_node_hook_index, blocks_end_node_order, blocks_end_node_name_hook, blocks_end_node_hook_input_or_ouput, blocks_end_node_hook_index
        
    def _join_remaining_layers_to_block(self):
       # 尽可能将剩余的层加入当前检测出的基本块中，方法是将两个块之间的节点尽可能加入前一个块，称之为最大待定块
        max_blocks = [] # list[list]结构，或者说list[block]，block = list[int]，每个块是块中包含的module和func在graph中的serial_number
        max_blocks_no_compressed_layers = [] # list[list]结构，对应于self.blocks，存储的是每个block中的非压缩层的name

        # 记录一个块中的开始节点是否为占位符，占位符分为两种情况： 
        # 一是该块的开始节点是其它块的终止节点，那么该块的开始节点中的操作就不再进行，如果为0则不是，如果为1则是占位符
        # 二是该块的开始节点是func操作，如cat, add等,则记为占位符
        max_blocks_start_node_is_placeholder: List[List[int]] = [] 
        
        max_blocks_start_node_order: List[List[int]] = [] # 记录每个块开始节点的编号
        max_blocks_start_node_name_hook: List[List] = [] # 记录每个块的输入应该钩原始模型中层的名字，每个按照编号顺序排列
        max_blocks_start_node_hook_input_or_ouput: List[List] = []  # 记录钩输入还是输出，如果为0则钩输入，如果为1则钩输出
        max_blocks_start_node_hook_index: List[List] = [] # 记录钩出的结果在hook的输入输出list的位置
        
        max_blocks_end_node_order: List[List[int]] = [] # 记录每个块终止节点的编号
        max_blocks_end_node_name_hook: List[List] = []  # 记录每个块的输出应该钩原始模型中层的名字
        max_blocks_end_node_hook_input_or_ouput: List[List] = []  # 记录钩输入还是输出，如果为0则钩输入，如果为1则钩输出
        max_blocks_end_node_hook_index: List[List] = [] # 记录钩出的结果在hook的输入输出list的位置
        
        
        all_node_order = list(self.graph.order_to_node.keys()) # 升序排列
        all_node_order.sort()
        # print(all_node_order)
        # exit(0)
        
        for index in range(len(self._blocks)):
            if index<len(self._blocks)-1:
                pre_block = self._blocks[index]
                next_block = self._blocks[index+1]
                max_block: List = copy.deepcopy(pre_block)
                print("第{}个块".format(index))
                print("pre block: {}".format(pre_block))
                print("next block: {}".format(next_block))
                min_order = max(pre_block)+1
                max_order = min(next_block)+1 if self._blocks_start_node_is_placeholder[index+1][0] else min(next_block)
                print("JOIN  min prder: {}, max order: {}".format(min_order, max_order))
            else:
                pre_block = self._blocks[index]
                max_block: List = copy.deepcopy(pre_block)
                print("第{}个块".format(index))
                print("pre block: {}".format(pre_block))
                print("next block: {}".format(None))
                min_order = max(pre_block)+1
                max_order = max(all_node_order)+1
                print("JOIN min prder: {}, max order: {}".format(min_order, max_order)) 
            
            for node_order in range(min_order, max_order):
                if node_order in all_node_order and node_order not in max_block:
                    max_block.append(node_order)

            max_block = list(set(max_block))
            max_block.sort()
            print("max block: {}".format(max_block))
            
            min_order = max(pre_block)-1
            max_order = max(max_block)
            
            print("REMOVE min prder: {}, max order: {}".format(min_order, max_order)) 
            
            find_block_flag = False
            for remove_node_order in range(max_order, min_order, -1):
                if remove_node_order not in max_block:
                    continue
                is_subgraph = self._adjust_is_subgraph(max_block)
                if is_subgraph:
                    is_block = self._adjust_is_block(max_block, max_blocks, compress_num_flag=False)
                else:
                    is_block = False
                print("is_subgraph: {}, is_block: {}".format(is_subgraph, is_block))
                # 发现一个满足条件的子图
                if is_subgraph and is_block:
                    start_node_list, end_node_list = self._find_all_start_and_end_nodes(max_block)
                    
                    # 判断开始节点是否为占位符，占位符的意思是该节点是上一个块的输出节点或者是func操作
                    start_nodes_is_placeholder = self._adjust_start_nodes_is_placeholder(start_node_list, max_blocks)
                    hook_tuple = self._adjust_block_hook_nodes_input_and_output(max_block, start_nodes_is_placeholder, start_node_list, end_node_list)
                    
                    assert len(hook_tuple)==6
                    block_able_train = True
                    for hook_list in hook_tuple:
                        for i in hook_list:
                            if i==-1:
                                block_able_train=False
                                break
                    
                    if block_able_train: 
                        # 当前块能被训练，且钩出当前块输入输出的节点

                        max_blocks_no_compressed_layers.append(self._find_block_all_paths_first_compressed_layer_name_from_end_nodes(max_block, end_node_list))

                        max_blocks_start_node_is_placeholder.append(start_nodes_is_placeholder)
                        max_blocks_start_node_order.append(start_node_list)
                        max_blocks_start_node_name_hook.append(hook_tuple[0])
                        max_blocks_start_node_hook_input_or_ouput.append(hook_tuple[1])
                        max_blocks_start_node_hook_index.append(hook_tuple[2])
                        
                        max_blocks_end_node_order.append(end_node_list)
                        max_blocks_end_node_name_hook.append(hook_tuple[3])
                        max_blocks_end_node_hook_input_or_ouput.append(hook_tuple[4])
                        max_blocks_end_node_hook_index.append(hook_tuple[5])
                        max_blocks.append(max_block)
                        print("final block: {}".format(max_block))
                        find_block_flag = True
                        break
                    
                max_block.remove(remove_node_order)     
                
            # pass

        self._blocks = max_blocks # list[list]结构，或者说list[block]，block = list[int]，每个块是块中包含的module和func在graph中的serial_number
        self._blocks_no_compressed_layers = max_blocks_no_compressed_layers # list[list]结构，对应于self.blocks，存储的是每个block中的非压缩层的name

        # 记录一个块中的开始节点是否为占位符，占位符分为两种情况： 
        # 一是该块的开始节点是其它块的终止节点，那么该块的开始节点中的操作就不再进行，如果为0则不是，如果为1则是占位符
        # 二是该块的开始节点是func操作，如cat, add等,则记为占位符
        self._blocks_start_node_is_placeholder = max_blocks_start_node_is_placeholder
        
        self._blocks_start_node_order = max_blocks_start_node_order # 记录每个块开始节点的编号
        self._blocks_start_node_name_hook = max_blocks_start_node_name_hook # 记录每个块的输入应该钩原始模型中层的名字，每个按照编号顺序排列
        self._blocks_start_node_hook_input_or_ouput = max_blocks_start_node_hook_input_or_ouput  # 记录钩输入还是输出，如果为0则钩输入，如果为1则钩输出
        self._blocks_start_node_hook_index = max_blocks_start_node_hook_index # 记录钩出的结果在hook的输入输出list的位置
        
        self._blocks_end_node_order = max_blocks_end_node_order # 记录每个块终止节点的编号
        self._blocks_end_node_name_hook = max_blocks_end_node_name_hook # 记录每个块的输出应该钩原始模型中层的名字
        self._blocks_end_node_hook_input_or_ouput = max_blocks_end_node_hook_input_or_ouput  # 记录钩输入还是输出，如果为0则钩输入，如果为1则钩输出
        self._blocks_end_node_hook_index = max_blocks_end_node_hook_index # 记录钩出的结果在hook的输入输出list的位置
       
    def detection_all_block(self):
        # 第一阶段，基础检测算法
        self.detection_all_basic_block()
        
        # 第二阶段: 尽可能将未加入块的卷积层加入块，即将两个块之间的节点尽可能加入前一个块
        self._join_remaining_layers_to_block()
        
    def print_blocks(self):
        assert len(self._blocks) == len(self._blocks_start_node_name_hook) == len(self._blocks_start_node_hook_input_or_ouput) == len(self._blocks_end_node_name_hook) == len(self._blocks_end_node_hook_input_or_ouput)

        num = 1
        for i, block in enumerate(self._blocks):
            assert len(self._blocks_start_node_is_placeholder[i]) == len(self._blocks_start_node_order[i]) == len(self._blocks_start_node_name_hook[i]) == len(self._blocks_start_node_hook_input_or_ouput[i]) == len(self._blocks_start_node_hook_index[i])

            assert len(self._blocks_end_node_order[i]) == len(self._blocks_end_node_name_hook[i]) == len(self._blocks_end_node_hook_input_or_ouput[i]) == len(self._blocks_end_node_hook_index[i])

            print("第{}个块, 有{}个输入节点，有{}个输出节点".format(num, len(self._blocks_start_node_name_hook[i]), len(self._blocks_end_node_name_hook[i])))
            
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
    block_detection = BaseBlockDetection(graph, min_compress_num=8, max_compress_num=16) # inceptionv3
    block_detection.print_reuse_layers()
    # exit(0)
    block_detection.detection_all_block()
    # print(block_detection.blocks)
    block_detection.print_blocks()
    # graph.show_graph()
