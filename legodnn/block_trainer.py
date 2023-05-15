import torch
import copy
import time
import os
import prettytable
import math
import shutil
import tqdm
from concurrent.futures import ThreadPoolExecutor

from .abstract_block_manager import AbstractBlockManager
from .abstract_model_manager import AbstractModelManager
from .utils.common.file import ensure_dir
from .utils.common.log import logger
# from .utils.common.log_1225 import get_logger
from .utils.dl.common.model import save_model, ModelSaveMethod

from legodnn.utils.dl.common.model import LayerActivation, get_module
from mmcv.parallel import MMDataParallel


class _BlockPack:
    def __init__(self, block, block_file_path, optimizer):
        self.block = block
        self.block_file_path = block_file_path
        self.optimizer = optimizer

        self.best_loss = 1e4
        self.last_loss = 0.0
        self.cur_loss = 0.0
        self.losses_record = []
        self.need_training = True
    

class BlockTrainer:
    def __init__(self, model, block_manager: AbstractBlockManager, model_manager: AbstractModelManager, 
                  compressed_blocks_dir, trained_blocks_dir,
                  max_epoch_num, train_loader,
                  optimizer=torch.optim.Adam([torch.zeros(1)], lr=3e-4), criterion=torch.nn.MSELoss(), alpha = 1.0,
                  worker_num=1, device='cuda', freeze_bn=False):
        
        self._block_manager = block_manager
        self._model_manager = model_manager
        self._model = model
        self._model.eval()
        self._pruned_blocks_dir = compressed_blocks_dir
        self._target_blocks_dir = trained_blocks_dir
        self._optimizer = optimizer
        self._alpha = alpha
        self._criterion = criterion
        self._max_epoch_num = max_epoch_num
        self._train_loader = train_loader
        self._worker_num = worker_num
        self._device = device

        self._freeze_bn = freeze_bn
        self._train_loss = None

    def _get_optimizers_of_all_pruned_blocks(self, all_pruned_blocks):
        optimizers = []
        # 遍历不同块的派生块集合 对于每一组派生块
        for pruned_blocks_in_same_loc in all_pruned_blocks:
            tmp_optimizers = []
            # 对于每一个不同稀疏度的派生块（视作一个模型） 都对应一个优化器
            for pruned_block in pruned_blocks_in_same_loc:
                tmp_optimizer = copy.deepcopy(self._optimizer)
                tmp_optimizer.add_param_group({'params': pruned_block[0].parameters()})
                tmp_optimizers += [tmp_optimizer]
            optimizers += [tmp_optimizers]
        # 最终得到 块数*稀疏度数 个优化器
        return optimizers

    def _get_block_pack_of_all_pruned_blocks(self):
        all_pruned_blocks_info = []
        # 遍历每个块
        for i, block_id in enumerate(self._block_manager.get_blocks_id()):
            pruned_blocks_info_in_same_loc = []
            # 遍历这个块稀疏度列表中的每个稀疏度 得到保存这些派生块的路径列表 并读取进来
            for block_sparsity in self._block_manager.get_blocks_sparsity()[i]:
                block_file_name = self._block_manager.get_block_file_name(block_id, block_sparsity)
                block_file_path = os.path.join(self._pruned_blocks_dir, block_file_name)
                pruned_blocks_info_in_same_loc += [(self._block_manager.get_block_from_file(block_file_path, self._device),
                                                    block_file_name)]
            all_pruned_blocks_info += [pruned_blocks_info_in_same_loc]
            # 最终得到一个列表[[b10 b11 b12 ...], [...], [...], ...]

        # 得到块列表对应的优化器列表
        optimizers = self._get_optimizers_of_all_pruned_blocks(all_pruned_blocks_info)
        res = []

        # 对于不同块的一组派生块和一组优化器
        for pruned_blocks_info_in_same_loc, optimizers_in_same_loc in zip(all_pruned_blocks_info, optimizers):
            tmp_res = []
            # 对于一个派生块和其对应的优化器 打包成一个训练环境（记录loss）
            for pruned_block_info, optimizer_of_it in zip(pruned_blocks_info_in_same_loc, optimizers_in_same_loc):
                target_block_file_path = os.path.join(self._target_blocks_dir, pruned_block_info[1])
                ensure_dir(target_block_file_path)
                tmp_res += [_BlockPack(pruned_block_info[0],
                                       target_block_file_path,
                                       optimizer_of_it)]
            res += [tmp_res]
        # 返回所有块打包好的训练环境
        return res

    def _backward_single_block(self, pruned_block_pack, data, target):
        try:
            # pruned_block_pack.optimizer.zero_grad()
            # data.requires_grad_(True)
            # pruned_block_output = pruned_block_pack.block(data)

            # if isinstance(pruned_block_output, tuple):
            #     loss = self._criterion(data[0], pruned_block_output[0])
            #     for batch_data, batch_target in (data[1:], pruned_block_output[1:]):
            #         loss += self._criterion(batch_data, batch_target)
            #     loss /= len(pruned_block_output)
            # else:
            #     loss = self._criterion(pruned_block_output, target)

            # pruned_block_pack.cur_loss += loss.item()
            # loss.backward()
            # pruned_block_pack.optimizer.step()
            pruned_block_pack.block.train()
            if self._freeze_bn:
                for m in pruned_block_pack.block.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        # print(m)
                        m.eval()
                
            pruned_block_pack.optimizer.zero_grad()
            if isinstance(data, tuple):
                for tuple_data in data:
                    tuple_data.requires_grad_(True)
            else:
                data.requires_grad_(True)
                
            pruned_block_output = pruned_block_pack.block(data)
            if isinstance(pruned_block_output, tuple):
                loss = self._alpha * self._criterion(target[0], pruned_block_output[0])
                for i in range(1, len(target)):
                    loss += self._alpha *self._criterion(target[i], pruned_block_output[i])
                loss /= len(pruned_block_output)
            else:
                loss = self._alpha * self._criterion(pruned_block_output, target)
            pruned_block_pack.cur_loss += loss.item()
            loss.backward()
            pruned_block_pack.optimizer.step()
            
        except Exception as e:
            logger.error(e)

    def  _save_model_frame(self):
        # model frame
        empty_model = copy.deepcopy(self._model)
        if isinstance(empty_model, MMDataParallel):
            empty_model = empty_model.module
        for block_id in self._block_manager.get_blocks_id():
            self._block_manager.empty_block_in_model(empty_model, block_id)
        model_frame_path = os.path.join(self._target_blocks_dir, 'model_frame.pt')
        ensure_dir(model_frame_path)
        print(empty_model)
        save_model(empty_model, model_frame_path, ModelSaveMethod.FULL)

    def train_all_blocks(self):
        self._save_model_frame()

        all_pruned_blocks_pack = self._get_block_pack_of_all_pruned_blocks()
        
        self._model.eval()
        self._model = self._model.to(self._device)
        raw_model = self._model
        if isinstance(raw_model, MMDataParallel):
            raw_model = raw_model.module
        # 得到每一组派生块的训练集 即原始块的输入输出
        # print("开始钩输入输出")
        # time.sleep(10)
        # name_to_la = self._block_manager.get_io_activation_of_all_modules(self._model, self._device)
        # name_to_la = self._block_manager.get_io_activation_of_all_blocks(self._model, self._device)
        name_to_la = self._block_manager.get_io_activation_of_all_blocks(raw_model, self._device)
        # print(name_to_la.keys())
        need_this_epoch = True
        # print("已经钩了输入输出")
        # time.sleep(10)
        logger.info('start block training...')

        # epoch start
        for epoch_index in range(self._max_epoch_num):
            if not need_this_epoch:
                logger.info('all blocks do not need training, early stop training')
                break

            epoch_start_time = time.time()

            # batch start
            batch_num = 0
            for batch_index, batch in tqdm.tqdm(enumerate(self._train_loader), total=len(self._train_loader), dynamic_ncols=True): # tmp modify for yolo
                
                batch_num += 1
                # 向前传播以获取中间数据
                self._model_manager.forward_to_gen_mid_data(self._model, batch, self._device)

                thread_pool = ThreadPoolExecutor(max_workers=self._worker_num)
                # each original block 对于一个块的训练集和训练环境
                for block_id, pruned_blocks_pack_in_same_loc in zip(self._block_manager.get_blocks_id(), all_pruned_blocks_pack):
                    # print("当前块的编号: {}".format(block_id))
                    
                    detection_manager = self._block_manager.detection_manager
                    # 得到输入数据
                    input_layer_activation_list = []
                    start_module_name_list = detection_manager.get_blocks_start_node_name_hook(block_id)
                    # print("当前需要钩的输入节点: {}".format(start_module_name_list))
                    
                    for start_module_name in start_module_name_list:
                        input_layer_activation_list.append(name_to_la.get(start_module_name))
                        
                    input_need_hook_input_list = []
                    start_node_hook_input_or_ouput_list = detection_manager.get_blocks_start_node_hook_input_or_ouput(block_id)
                    for start_node_hook_input_or_ouput in start_node_hook_input_or_ouput_list:
                        input_need_hook_input_list.append(start_node_hook_input_or_ouput == 0)
                    start_hook_index_list = detection_manager.get_blocks_start_node_hook_index(block_id)
                    input_data = ()
                    # print("当前输入在钩出数据中的位置: {}".format(start_hook_index_list))
                    
                    for i, layer_activation in enumerate(input_layer_activation_list):
                        # print("输入节点{}，钩出输入的长度{}，索引位置{}".format(start_module_name_list[i], len(layer_activation.input_list), start_hook_index_list[i]))
                        input_data = input_data + (layer_activation.input_list[start_hook_index_list[i]] if input_need_hook_input_list[i] else layer_activation.output_list[start_hook_index_list[i]],)
                        
                    if len(start_module_name_list) == 1:
                        input_data = input_data[0]

                    # 得到输出数据
                    output_layer_activation_list = []
                    end_module_name_list = detection_manager.get_blocks_end_node_name_hook(block_id)
                    # print("当前需要钩的输出节点: {}".format(end_module_name_list))
                    for end_module_name in end_module_name_list:
                        output_layer_activation_list.append(name_to_la.get(end_module_name))
                    output_need_hook_input_list = []
                    end_node_hook_input_or_ouput_list = detection_manager.get_blocks_end_node_hook_input_or_ouput(block_id)
                    for end_node_hook_input_or_ouput in end_node_hook_input_or_ouput_list:
                        output_need_hook_input_list.append(end_node_hook_input_or_ouput == 0)
                    end_hook_index_list = detection_manager.get_blocks_end_node_hook_index(block_id)
                    output_data = ()
                    # print("当前输出在钩出数据中的位置: {}".format(start_hook_index_list))
                    for i, layer_activation in enumerate(output_layer_activation_list):
                        # print("输出节点{}，钩出输出的长度{}，索引位置{}".format(start_module_name_list[i], len(layer_activation.input_list), start_hook_index_list[i]))
                        output_data = output_data + (layer_activation.input_list[end_hook_index_list[i]] if output_need_hook_input_list[i] else layer_activation.output_list[end_hook_index_list[i]],)
                        
                    if len(end_module_name_list) == 1:
                        output_data = output_data[0]
                        
                    # for input_need_hook_input, output_need_hook_input, start_module_name, end_module_name in zip(input_need_hook_input_list, output_need_hook_input_list, start_module_name_list, end_module_name_list):
                    #     print('正在训练块{}, 当前块的输入是层{}的{}, 当前块的输出是层{}的{}'.format(block_id, start_module_name,
                    #     '输入' if input_need_hook_input else '输出', end_module_name, '输入' if output_need_hook_input else '输出'))
                    #     print('输入形状{}, 输出形状{}'.format(input_data.size(), output_data.size()))

                    # each pruned block of corresponding original block 对于相同块的不同稀疏度的块的训练环境
                    for pruned_block_index, pruned_block_pack in enumerate(pruned_blocks_pack_in_same_loc):
                        # print('第{}个稀疏度的块:'.format(pruned_block_index))
                        # print(pruned_block_pack.block)
                        # save_model(pruned_block_pack.block, '/data/gxy/legodnn-public-version_9.27/blocks_test/' + str(block_index) + '_' + str(pruned_block_index), 
                        # ModelSaveMethod.ONNX, input_data.size())
                        if not pruned_block_pack.need_training:
                            continue

                        thread_pool.submit(self._backward_single_block, pruned_block_pack, input_data, output_data)
                        
                        # pruned_block_pack.optimizer.zero_grad()
                        # input_data.requires_grad_(True)
                        # pruned_block_output = pruned_block_pack.block(input_data)
                        # loss = self._criterion(pruned_block_output, output_data)

                        # pruned_block_pack.cur_loss += loss.item()
                        # loss.backward()
                        # pruned_block_pack.optimizer.step()

                        # pruned_block_pack.optimizer.zero_grad()
                        # if isinstance(input_data, tuple):
                        #     for tuple_data in input_data:
                        #         tuple_data.requires_grad_(True)
                        # else:
                        #     input_data.requires_grad_(True)
                            
                        # pruned_block_output = pruned_block_pack.block(input_data)
                        # # print("输入数据形状{}".format(input_data.size()))
                        # # print("原始块输出数据形状".format(output_data.size()))
                        # # print("输出数据形状{}".format(pruned_block_output.size()))
                        # if isinstance(pruned_block_output, tuple):
                        #     loss = self._criterion(output_data[0], pruned_block_output[0])
                        #     # for batch_data, batch_target in zip(output_data[1:], pruned_block_output[1:]):
                        #     #     loss += self._criterion(batch_data, batch_target)
                        #     for i in range(1, len(output_data)):
                        #         loss += self._criterion(output_data[i], pruned_block_output[i])
                        #     loss /= len(pruned_block_output)
                        # else:
                        #     loss = self._criterion(pruned_block_output, output_data)

                        # pruned_block_pack.cur_loss += loss.item()
                        # # print(loss.requires_grad)
                        # loss.backward()
                        # pruned_block_pack.optimizer.step()

                thread_pool.shutdown(wait=True)
                # print(11111)
                # 这句话很重要，释放勾出的输入输出，防止显存爆照，否则，输入输出会因为list.append()一直累积
                self._block_manager.clear_io_activations(name_to_la)
                # self._block_manager.clear_io_activation_of_all_modules()
            # batch end

            # update info

            need_training_blocks_num = 0
            need_this_epoch = False
            table_str = ''

            for i, pruned_blocks_pack_in_same_loc in enumerate(all_pruned_blocks_pack):

                loss_table = prettytable.PrettyTable()
                blocks_id = self._block_manager.get_blocks_id()
                loss_table.field_names = [''] + self._block_manager.get_blocks_sparsity()[i]
                row = [blocks_id[i]]

                for j, pruned_block_pack in enumerate(pruned_blocks_pack_in_same_loc):

                    if pruned_block_pack.need_training:
                        pruned_block_pack.cur_loss /= batch_num   # ???

                    if pruned_block_pack.cur_loss < pruned_block_pack.best_loss:
                        pruned_block_pack.best_loss = pruned_block_pack.cur_loss
                        save_model(pruned_block_pack.block, pruned_block_pack.block_file_path, ModelSaveMethod.FULL)

                    if pruned_block_pack.last_loss == 0 or not pruned_block_pack.need_training:
                        row += ['{:.8f}\n(-)'.format(pruned_block_pack.cur_loss)]
                    else:
                        row += ['{:.8f}\n(↓ {:.8f})'.format(pruned_block_pack.cur_loss,
                                                            pruned_block_pack.last_loss - pruned_block_pack.cur_loss)]

                    if epoch_index > 0 and \
                            not self._block_manager.should_continue_train_block(pruned_block_pack.last_loss,
                                                                                pruned_block_pack.cur_loss):
                        pruned_block_pack.need_training = False
                    else:
                        need_this_epoch = True
                        need_training_blocks_num += 1

                    if pruned_block_pack.need_training:
                        pruned_block_pack.last_loss = pruned_block_pack.cur_loss
                        pruned_block_pack.losses_record += [pruned_block_pack.cur_loss]
                        pruned_block_pack.cur_loss = 0.0
                        
                        # with open(pruned_block_pack.block_file_path + '.loss-record', 'w') as f:
                        #     f.write(json.dumps(pruned_block_pack.losses_record))

                loss_table.add_row(row)
                table_str += str(loss_table) + '\n'

            logger.info('epoch {} ({:.6f}s, {} blocks still need training), '
                        'blocks loss: \n{}'.format(epoch_index, time.time() - epoch_start_time,
                                                   need_training_blocks_num, table_str))

        # epoch end

        # [o.remove() for o in original_blocks_io_activation]
        [la.remove() for la in name_to_la.values()]
