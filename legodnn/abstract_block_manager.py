import abc
import torch.nn

from typing import List, Any, Tuple

from .abstract_model_manager import AbstractModelManager


class AbstractBlockManager(abc.ABC):
    """Define all attributes of LegoDNN blocks.
    """
    
    def __init__(self, blocks_id: List[str], blocks_sparsity: List[List[float]], model_manager: AbstractModelManager):
        """
        Args:
            blocks_id (List[str]): 
                The id of LegoDNN blocks which you want to generate, fine-tune, profile and so on. 
                The id name depends on yourself, make sure that each LegoDNN block has an unique id.
            blocks_sparsity (List[List[float]]): The target sparsity of corresponding LegoDNN blocks.
            model_manager (AbstractModelManager): The manager which defines all attributes of a model \
                which is used to generate LegoDNN blocks.
        """
        self._blocks_id = blocks_id
        self._blocks_sparsity = blocks_sparsity
        self._model_manager = model_manager

    def get_blocks_id(self):
        """Return the id of LegoDNN blocks.

        Returns:
            List[str]: The id of LegoDNN blocks.
        """
        return self._blocks_id

    def get_blocks_sparsity(self):
        """Return the target sparsity of LegoDNN blocks.

        Returns:
            List[List[float]]: The target sparsity of LegoDNN blocks.
        """
        return self._blocks_sparsity
    
    @abc.abstractclassmethod
    def get_default_blocks_id():
        """Return the id of ALL LegoDNN blocks. Differ from :method:`get_blocks_id()`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_block_from_model(self, model: torch.nn.Module, block_id: str):
        """Get the (shallow-copied, e.g. reference) corresponding block from the given PyTorch model by block id.

        Args:
            model (torch.nn.Module): Given PyTorch model.
            block_id (str): Id of target block.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_block_to_model(self, model: torch.nn.Module, block_id: str, block: torch.nn.Module):
        """Set the given block in the corresponding location of the given model.

        Args:
            model (torch.nn.Module): Given PyTorch model.
            block_id (str): The id of given block.
            block (torch.nn.Module): Given block.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def empty_block_in_model(self, model: torch.nn.Module, block_id: str):
        """Empty the corresponding location of the given model.

        Args:
            model (torch.nn.Module): Given PyTorch model
            block_id (str): The id of the location.
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_pruned_block(self, model: torch.nn.Module, block_id: str, block_sparsity: float, 
                         model_input_size: Tuple[int], device: str):
        """Get the pruned block of the corresponding block in the given model.
        The pruning process shouldn't mutate the given model.
        And this function should return the deep-copied block which has no relation with the given model.
        
        Args:
            model (torch.nn.Module): Given PyTorch model.
            block_id (str): The id of target block.
            block_sparsity (float): Target sparsity of pruned block.
            model_input_size (Tuple[int]): Typically be `(1, 3, 32, 32)` or `(1, 3, 224, 224)`.
            device (str): Where the pruning process is located.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_io_activation_of_all_blocks(self, model: torch.nn.Module, device: str):
        """Get instances of :class:`LayerActivation` or :class:`LayerActivationWrapper` of blocks.

        Args:
            model (torch.nn.Module): Given PyTorch model.
            device (str): Typically be 'cpu' or 'cuda'.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_time_profilers_of_all_blocks(self, model: torch.nn.Module, device: str):
        """Get instances of :class:`TimeProfiler` or :class:`TimeProfilerWrapper` of blocks.

        Args:
            model (torch.nn.Module): Given PyTorch model.
            device (str): Typically be 'cpu' or 'cuda'.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_block_file_name(self, block_id: str, block_sparsity: float):
        """Get the unique file name of the block via its id and sparsity.

        Args:
            block_id (str): The id of target block.
            block_sparsity (float): Target sparsity of pruned block.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def save_block_to_file(self, block: torch.nn.Module, block_file_path: str):
        """Save the given block to the disk.

        Args:
            block (torch.nn.Module): Given block.
            block_file_path (str): The location for saving the block.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_block_from_file(self, block_file_path: str, device: str):
        """Load block from file which generated by :attr:`save_block_to_file`.

        Args:
            block_file_path (str): The location for saving the block.
            device (str): Typically be 'cpu' or 'cuda'.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def should_continue_train_block(self, last_loss: float, cur_loss: float):
        """Whether to continue training a block. Return True to continue.

        Args:
            last_loss (float): Loss value in last epoch.
            cur_loss (float): Current loss value.
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_block_size(self, block: torch.nn.Module):
        """Get the size of the block.

        Args:
            block (torch.nn.Module): Give block.
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_block_flops_and_params(self, block: torch.nn.Module, input_size: Tuple[int]):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_block_latency(self, block, sample_num, input_size, device):
        raise NotImplementedError()
