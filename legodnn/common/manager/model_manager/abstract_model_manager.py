import abc
from typing import Tuple
import torch
from torch.utils.data import DataLoader


class AbstractModelManager(abc.ABC):
    """Define all attributes of the model.
    """
    
    @abc.abstractmethod
    def forward_to_gen_mid_data(self, model: torch.nn.Module, batch_data: Tuple, device: str):
        """Let model perform an inference on given data.

        Args:
            model (torch.nn.Module): A PyTorch model.
            batch_data (Tuple): A batch of data, typically be `(data, target)`.
            device (str): Typically be 'cpu' or 'cuda'.
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def dummy_forward_to_gen_mid_data(self, model: torch.nn.Module, model_input_size: Tuple[int], device: str):
        """Let model perform a dummy inference.

        Args:
            model (torch.nn.Module): A PyTorch model.
            model_input_size (Tuple[int]): Typically be `(1, 3, 32, 32)` or `(1, 3, 224, 224)`.
            device (str): Typically be 'cpu' or 'cuda'.
        """
        raise NotImplementedError()
    
    @abc.abstractmethod 
    def get_model_acc(self, model: torch.nn.Module, test_loader: DataLoader, device: str):
        """Get the test accuracy of the model.

        Args:
            model (torch.nn.Module): A PyTorch model.
            test_loader (DataLoader): Test data loader.
            device (str): Typically be 'cpu' or 'cuda'.
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_model_size(self, model: torch.nn.Module):
        """Get the size of the model file (in byte).

        Args:
            model (torch.nn.Module): A PyTorch model.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_model_flops_and_param(self, model: torch.nn.Module, model_input_size: Tuple[int]):
        """Get the FLOPs and the number of parameters of the model, return as (FLOPs, param).

        Args:
            model (torch.nn.Module): A PyTorch model.
            model_input_size (Tuple[int]): Typically be `(1, 3, 32, 32)` or `(1, 3, 224, 224)`.
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_model_latency(self, model: torch.nn.Module, sample_num: int, model_input_size: Tuple[int], device: str):
        """Get the inference latency of the model.

        Args:
            model (torch.nn.Module): A PyTorch model.
            sample_num (int): How many samples is used in the test.
            model_input_size (Tuple[int]): Typically be `(1, 3, 32, 32)` or `(1, 3, 224, 224)`.
            device (str): Typically be 'cpu' or 'cuda'.
        """
        raise NotImplementedError()
