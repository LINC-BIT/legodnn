# import copy
# import time
# import os
# import prettytable
# import tqdm
# import torch
# from concurrent.futures import ThreadPoolExecutor

# from .abstract_block_manager import AbstractBlockManager
# from .abstract_model_manager import AbstractModelManager
# from .utils.common.file import ensure_dir
# from .utils.common.log import logger
# from .utils.dl.common.model import save_model, ModelSaveMethod


# class _BlockPack:
#     def __init__(self, block, block_file_path, optimizer):
#         self.block = block
#         self.block_file_path = block_file_path
#         self.optimizer = optimizer

#         self.best_loss = 1e4
#         self.last_loss = 0.0
#         self.cur_loss = 0.0
#         self.need_training = True


# class BlockTrainer:
#     def __init__(self, model, block_manager: AbstractBlockManager, model_manager: AbstractModelManager,
#                  compressed_blocks_dir, trained_blocks_dir,
#                  max_epoch_num, train_loader,
#                  optimizer=torch.optim.Adam([torch.zeros(1)], lr=3e-4), criterion=torch.nn.MSELoss(),
#                  worker_num=1, device='cuda'):

#         self._block_manager = block_manager
#         self._model_manager = model_manager
#         self._model = model
#         self._compressed_blocks_dir = compressed_blocks_dir
#         self._trained_blocks_dir = trained_blocks_dir
#         self._optimizer = optimizer
#         self._criterion = criterion
#         self._max_epoch_num = max_epoch_num
#         self._train_loader = train_loader
#         self._worker_num = worker_num
#         self._device = device

#         self._train_loss = None

#     def _get_optimizers_of_all_compressed_blocks(self, all_compressed_blocks):
#         optimizers = []
#         for compressed_blocks_in_same_loc in all_compressed_blocks:
#             tmp_optimizers = []
#             for compressed_block in compressed_blocks_in_same_loc:
#                 tmp_optimizer = copy.deepcopy(self._optimizer)
#                 tmp_optimizer.add_param_group({'params': compressed_block[0].parameters()})
#                 tmp_optimizers += [tmp_optimizer]
#             optimizers += [tmp_optimizers]

#         return optimizers

#     def _get_block_pack_of_all_compressed_blocks(self):
#         all_compressed_blocks_info = []
#         for i, block_id in enumerate(self._block_manager.get_blocks_id()):
#             compressed_blocks_info_in_same_loc = []
#             for block_sparsity in self._block_manager.get_blocks_sparsity()[i]:
#                 block_file_name = self._block_manager.get_block_file_name(block_id, block_sparsity)
#                 block_file_path = os.path.join(self._compressed_blocks_dir, block_file_name)
#                 compressed_blocks_info_in_same_loc += [(self._block_manager.get_block_from_file(block_file_path, self._device),
#                                                     block_file_name)]
#             all_compressed_blocks_info += [compressed_blocks_info_in_same_loc]

#         optimizers = self._get_optimizers_of_all_compressed_blocks(all_compressed_blocks_info)
#         res = []

#         for compressed_blocks_info_in_same_loc, optimizers_in_same_loc in zip(all_compressed_blocks_info, optimizers):
#             tmp_res = []
#             for compressed_block_info, optimizer_of_it in zip(compressed_blocks_info_in_same_loc, optimizers_in_same_loc):
#                 target_block_file_path = os.path.join(self._trained_blocks_dir, compressed_block_info[1])
#                 ensure_dir(target_block_file_path)
#                 tmp_res += [_BlockPack(compressed_block_info[0],
#                                        target_block_file_path,
#                                        optimizer_of_it)]
#             res += [tmp_res]

#         return res

#     def _backward_single_block(self, compressed_block_pack, data, target):
#         # torch.cuda.synchronize()

#         stream = torch.cuda.Stream()
#         with torch.cuda.stream(stream):
#             # try:
#             compressed_block_pack.optimizer.zero_grad()
#             data.requires_grad_(True)
#             compressed_block_output = compressed_block_pack.block(data)
#             loss = self._criterion(compressed_block_output, target)

#             # print(loss)

#             compressed_block_pack.cur_loss += loss.item()
#             loss.backward()
#             compressed_block_pack.optimizer.step()
#             # except Exception as e:
#             #     logger.error(e)

#         # torch.cuda.synchronize()

#     def _save_model_frame(self):
#         # model frame
#         empty_model = copy.deepcopy(self._model)
#         for block_id in self._block_manager.get_blocks_id():
#             self._block_manager.empty_block_in_model(empty_model, block_id)
#         model_frame_path = os.path.join(self._trained_blocks_dir, 'model_frame.pt')
#         ensure_dir(model_frame_path)
#         save_model(empty_model, model_frame_path, ModelSaveMethod.FULL)

#     def train_all_blocks(self):
#         self._save_model_frame()

#         all_compressed_blocks_pack = self._get_block_pack_of_all_compressed_blocks()

#         self._model.eval()
#         self._model = self._model.to(self._device)
#         original_blocks_io_activation = self._block_manager.get_io_activation_of_all_blocks(self._model, self._device)

#         # print(str(original_blocks_io_activation[2]))
#         # print('---')
#         # print(str(original_blocks_io_activation[3]))

#         # print('-----')
#         b1 = copy.deepcopy(all_compressed_blocks_pack[2][0].block)
#         b2 = copy.deepcopy(self._block_manager.get_block_from_model(self._model, 'g-2'))

#         def equal_block(b1, b2):
#             di = torch.rand((1, 100, 1, 1)).cuda()
#             with torch.no_grad():
#                 print(b1(di).equal(b2(di)))
#             # assert len(b1.state_dict().keys()) == len(b2.state_dict().keys())
#             # print('--')
#             # for k, v1, v2 in zip(b1.state_dict().keys(), b1.state_dict().values(), b2.state_dict().values()):
#             #     print(k, v1.equal(v2))
#             # print('--')

#         equal_block(b1, b2)

#         # di = torch.rand((1, 100, 1, 1)).cuda()
#         # print(b1(di) == b2(di))
#         # print()
#         # print('---')
#         # print(all_compressed_blocks_pack[3][0].block)

#         # print(self._block_manager.get_block_from_model(self._model, 'g-2'))

#         need_this_epoch = True

#         logger.info('start block training...')

#         # epoch start
#         for epoch_index in range(self._max_epoch_num):
#             if not need_this_epoch:
#                 logger.info('all blocks do not need training, early stop training')
#                 break

#             epoch_start_time = time.time()

#             # batch start
#             batch_num = 0
#             for batch_index, batch in tqdm.tqdm(enumerate(self._train_loader), total=len(self._train_loader), dynamic_ncols=True):
#                 batch_num += 1

#                 # b3 = copy.deepcopy(all_compressed_blocks_pack[2][0].block)
#                 # b4 = copy.deepcopy(self._block_manager.get_block_from_model(self._model, 'g-2'))
#                 # equal_block(b3, b4)

#                 self._model_manager.forward_to_gen_mid_data(self._model, batch, self._device)

#                 # b3 = copy.deepcopy(all_compressed_blocks_pack[2][0].block)
#                 # b4 = copy.deepcopy(self._block_manager.get_block_from_model(self._model, 'g-2'))
#                 # equal_block(b3, b4)

#                 thread_pool = ThreadPoolExecutor(max_workers=self._worker_num)

#                 # each original block
#                 for block_index, (original_block_io_activation, compressed_blocks_pack_in_same_loc) in \
#                         enumerate(zip(original_blocks_io_activation, all_compressed_blocks_pack)):

#                     # each compressed block of corresponding original block
#                     for compressed_block_index, compressed_block_pack in enumerate(compressed_blocks_pack_in_same_loc):

#                         if not compressed_block_pack.need_training:
#                             continue

#                         # thread_pool.submit(self._backward_single_block, compressed_block_pack,
#                         #                    original_block_io_activation.input,
#                         #                    original_block_io_activation.output)
#                         # b3 = copy.deepcopy(all_compressed_blocks_pack[2][0].block)
#                         # b4 = copy.deepcopy(self._block_manager.get_block_from_model(self._model, 'g-2'))
#                         # equal_block(b3, b4)

#                         self._backward_single_block(compressed_block_pack,
#                                            original_block_io_activation.input,
#                                            original_block_io_activation.output)

#                         # b3 = copy.deepcopy(all_compressed_blocks_pack[2][0].block)
#                         # b4 = copy.deepcopy(self._block_manager.get_block_from_model(self._model, 'g-2'))
#                         # equal_block(b3, b4)

#                 # thread_pool.shutdown(wait=True) # ztrace: ignore
#             # batch end

#             # update loss table
#             need_training_blocks_num = 0
#             need_this_epoch = False
#             table_str = ''

#             for i, compressed_blocks_pack_in_same_loc in enumerate(all_compressed_blocks_pack):

#                 loss_table = prettytable.PrettyTable()
#                 blocks_id = self._block_manager.get_blocks_id()
#                 loss_table.field_names = [''] + self._block_manager.get_blocks_sparsity()[i]
#                 row = [blocks_id[i]]

#                 for j, compressed_block_pack in enumerate(compressed_blocks_pack_in_same_loc):

#                     if compressed_block_pack.need_training:
#                         compressed_block_pack.cur_loss /= batch_num

#                     if compressed_block_pack.cur_loss < compressed_block_pack.best_loss:
#                         compressed_block_pack.best_loss = compressed_block_pack.cur_loss
#                         save_model(compressed_block_pack.block, compressed_block_pack.block_file_path, ModelSaveMethod.FULL)

#                     if compressed_block_pack.last_loss == 0 or not compressed_block_pack.need_training:
#                         row += ['{:.8f}\n(-)'.format(compressed_block_pack.cur_loss)]
#                     else:
#                         row += ['{:.8f}\n(↓ {:.8f})'.format(compressed_block_pack.cur_loss,
#                                                             compressed_block_pack.last_loss - compressed_block_pack.cur_loss)]

#                     if epoch_index > 0 and \
#                             not self._block_manager.should_continue_train_block(compressed_block_pack.last_loss,
#                                                                                 compressed_block_pack.cur_loss):
#                         compressed_block_pack.need_training = False
#                     else:
#                         need_this_epoch = True
#                         need_training_blocks_num += 1

#                     if compressed_block_pack.need_training:
#                         compressed_block_pack.last_loss = compressed_block_pack.cur_loss
#                         compressed_block_pack.cur_loss = 0.0 # ztrace: ignore

#                 loss_table.add_row(row)
#                 table_str += str(loss_table) + '\n'

#             logger.info('epoch {} ({:.6f}s, {} blocks still need training), '
#                         'blocks loss: \n{}'.format(epoch_index, time.time() - epoch_start_time,
#                                                    need_training_blocks_num, table_str))

#         # epoch end


# from legodnn.manager import BlockManager, ModelManager
# from utils.model import *
# from utils.utils import *
import torch
import copy
import time
import os
import prettytable
import tqdm
from concurrent.futures import ThreadPoolExecutor

from legodnn.common.manager import AbstractBlockManager
from legodnn.common.manager import AbstractModelManager
from legodnn.common.utils.common.file import ensure_dir
from legodnn.common.utils import logger
from legodnn.common.utils import save_model, ModelSaveMethod


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
                  optimizer=torch.optim.Adam([torch.zeros(1)], lr=3e-4), criterion=torch.nn.MSELoss(),
                  worker_num=1, device='cuda'):

        self._block_manager = block_manager
        self._model_manager = model_manager
        self._model = model
        self._pruned_blocks_dir = compressed_blocks_dir
        self._target_blocks_dir = trained_blocks_dir
        self._optimizer = optimizer
        self._criterion = criterion
        self._max_epoch_num = max_epoch_num
        self._train_loader = train_loader
        self._worker_num = worker_num
        self._device = device

        self._train_loss = None

    def _get_optimizers_of_all_pruned_blocks(self, all_pruned_blocks):
        optimizers = []
        for pruned_blocks_in_same_loc in all_pruned_blocks:
            tmp_optimizers = []
            for pruned_block in pruned_blocks_in_same_loc:
                tmp_optimizer = copy.deepcopy(self._optimizer)
                tmp_optimizer.add_param_group({'params': pruned_block[0].parameters()})
                tmp_optimizers += [tmp_optimizer]
            optimizers += [tmp_optimizers]

        return optimizers

    def _get_block_pack_of_all_pruned_blocks(self):
        all_pruned_blocks_info = []
        for i, block_id in enumerate(self._block_manager.get_blocks_id()):
            pruned_blocks_info_in_same_loc = []
            for block_sparsity in self._block_manager.get_blocks_sparsity()[i]:
                block_file_name = self._block_manager.get_block_file_name(block_id, block_sparsity)
                block_file_path = os.path.join(self._pruned_blocks_dir, block_file_name)
                pruned_blocks_info_in_same_loc += [(self._block_manager.get_block_from_file(block_file_path, self._device),
                                                    block_file_name)]
            all_pruned_blocks_info += [pruned_blocks_info_in_same_loc]

        optimizers = self._get_optimizers_of_all_pruned_blocks(all_pruned_blocks_info)
        res = []

        for pruned_blocks_info_in_same_loc, optimizers_in_same_loc in zip(all_pruned_blocks_info, optimizers):
            tmp_res = []
            for pruned_block_info, optimizer_of_it in zip(pruned_blocks_info_in_same_loc, optimizers_in_same_loc):
                target_block_file_path = os.path.join(self._target_blocks_dir, pruned_block_info[1])
                ensure_dir(target_block_file_path)
                tmp_res += [_BlockPack(pruned_block_info[0],
                                       target_block_file_path,
                                       optimizer_of_it)]
            res += [tmp_res]

        return res

    def _backward_single_block(self, pruned_block_pack, data, target):
        try:
            pruned_block_pack.optimizer.zero_grad()
            data.requires_grad_(True)
            pruned_block_output = pruned_block_pack.block(data)
            loss = self._criterion(pruned_block_output, target)

            pruned_block_pack.cur_loss += loss.item()
            loss.backward()
            pruned_block_pack.optimizer.step()
        except Exception as e:
            logger.error(e)

    def  _save_model_frame(self):
        # model frame
        empty_model = copy.deepcopy(self._model)
        for block_id in self._block_manager.get_blocks_id():
            self._block_manager.empty_block_in_model(empty_model, block_id)
        model_frame_path = os.path.join(self._target_blocks_dir, 'model_frame.pt')
        ensure_dir(model_frame_path)
        save_model(empty_model, model_frame_path, ModelSaveMethod.FULL)

    def train_all_blocks(self):
        self._save_model_frame()

        all_pruned_blocks_pack = self._get_block_pack_of_all_pruned_blocks()

        self._model.eval()
        self._model = self._model.to(self._device)
        original_blocks_io_activation = self._block_manager.get_io_activation_of_all_blocks(self._model, self._device)

        need_this_epoch = True

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
                # Yolov3: (data, target, paths, _) && data = data.float() / 255.0
                # normal: (data, target)
                # data, target = self._get_data_and_target(batch)

                batch_num += 1

                # inference in original model
                # data = data.float() / 255.0
                # data, target = data.to(self._device), target.to(self._device)
                # with torch.no_grad():
                #     self._model(data)
                self._model_manager.forward_to_gen_mid_data(self._model, batch, self._device)

                thread_pool = ThreadPoolExecutor(max_workers=self._worker_num)

                # each original block
                for block_index, (original_block_io_activation, pruned_blocks_pack_in_same_loc) in \
                        enumerate(zip(original_blocks_io_activation, all_pruned_blocks_pack)):

                    # each pruned block of corresponding original block
                    for pruned_block_index, pruned_block_pack in enumerate(pruned_blocks_pack_in_same_loc):

                        if not pruned_block_pack.need_training:
                            continue

                        thread_pool.submit(self._backward_single_block, pruned_block_pack,
                                           original_block_io_activation.input,
                                           original_block_io_activation.output)
                        # self._backward_single_block(pruned_block_pack, original_block_io_activation.input,
                        #                             original_block_io_activation.output)

                thread_pool.shutdown(wait=True)
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
                        pruned_block_pack.cur_loss /= batch_num

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

        [o.remove() for o in original_blocks_io_activation]
