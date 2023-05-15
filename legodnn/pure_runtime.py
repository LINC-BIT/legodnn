import torch
import os

from .abstract_block_manager import AbstractBlockManager
from .utils.common.log import logger


class PureRuntime:
    def __init__(self, blocks_dir, block_manager: AbstractBlockManager, device):
        self._blocks_dir = blocks_dir
        self._block_manager = block_manager
        self._device = device

        self._model = torch.load(os.path.join(blocks_dir, 'model_frame.pt'), map_location='cpu').to(device)
        # if hasattr(self._model, 'forward_dummy'):
        #     self._model.forward = self._model.forward_dummy
            
        self._blocks_name = self._block_manager.get_blocks_id()
        self._cur_blocks_sparsity = [-1 for _ in range(len(self._blocks_name))]

    def get_model(self):
        return self._model

    def load_blocks(self, blocks_sparsity, return_cost=False):
        assert len(blocks_sparsity) == len(self._blocks_name)
        logger.info('load blocks with sparsity {}'.format(blocks_sparsity))

        page_in_size, page_out_size = 0, 0

        for i, (block_name, block_sparsity, cur_block_sparsity) in enumerate(zip(self._blocks_name, blocks_sparsity,
                                                                  self._cur_blocks_sparsity)):
            if block_sparsity == cur_block_sparsity:
                continue

            block = self._block_manager.get_block_from_file(os.path.join(self._blocks_dir, 
                                                                         self._block_manager.get_block_file_name(block_name, block_sparsity)), self._device)
            if return_cost:
                page_out_size += self._block_manager.get_block_size(self._block_manager.get_block_from_model(self._model, block_name))
            self._block_manager.set_block_to_model(self._model, block_name, block)
            if return_cost:
                page_in_size += self._block_manager.get_model_size(self._block_manager.get_block_from_model(self._model, block_name))
            logger.debug('load {}th block ({}) (sparsity {}) from file'.format(i, block_name, block_sparsity))

        self._cur_blocks_sparsity = blocks_sparsity

        if return_cost:
            return page_in_size, page_out_size

    def empty_blocks(self, return_cost=False):
        page_out_size = 0
        for block_id in self._block_manager.get_blocks_id():
            if return_cost:
                before_block_size = self._block_manager.get_model_size(self._block_manager.get_block_from_model(self._model, block_id))
            self._block_manager.empty_block_in_model(self._model, block_id)
            if return_cost:
                page_out_size += before_block_size - self._block_manager.get_model_size(self._block_manager.get_block_from_model(self._model, block_id))

        logger.info('empty blocks in model')
        if return_cost:
            return page_out_size
