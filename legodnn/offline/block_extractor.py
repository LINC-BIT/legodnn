from legodnn.common.manager import AbstractBlockManager
from legodnn.common.utils import logger,ensure_dir,save_model,ModelSaveMethod

import copy
import os


class BlockExtractor:
    def __init__(self, model, block_manager: AbstractBlockManager,
                 blocks_saved_dir, model_input_size,
                 device):

        self._model = model
        self._block_manager = block_manager
        self._blocks_saved_dir = blocks_saved_dir
        self._dummy_input_size = model_input_size
        self._device = device

    def _save_compressed_block(self, model, block_id, block_sparsity):
        compressed_block = self._block_manager.get_pruned_block(model, block_id, block_sparsity,
                                                            self._dummy_input_size, self._device)

        pruned_block_file_path = os.path.join(self._blocks_saved_dir,
                                              self._block_manager.get_block_file_name(block_id, block_sparsity))
        self._block_manager.save_block_to_file(compressed_block, pruned_block_file_path)
        logger.info('save pruned block {} (sparsity {}) in {}'.format(block_id, block_sparsity, pruned_block_file_path))
        logger.debug(compressed_block)

    def _compress_single_block(self, block_id, block_sparsity):
        model = copy.deepcopy(self._model)
        model = model.to(self._device)
        self._save_compressed_block(model, block_id, block_sparsity)
        
    def _save_model_frame(self):
        # model frame
        empty_model = copy.deepcopy(self._model)
        for block_id in self._block_manager.get_blocks_id():
            self._block_manager.empty_block_in_model(empty_model, block_id)
        model_frame_path = os.path.join(self._blocks_saved_dir, 'model_frame.pt')
        ensure_dir(model_frame_path)
        save_model(empty_model, model_frame_path, ModelSaveMethod.FULL)

    def extract_all_blocks(self):
        self._save_model_frame()

        for i, block_id in enumerate(self._block_manager.get_blocks_id()):
            for block_sparsity in self._block_manager.get_blocks_sparsity()[i]:
                self._compress_single_block(block_id, block_sparsity)
