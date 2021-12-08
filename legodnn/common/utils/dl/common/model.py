import enum
import time
from typing import List, Tuple, Type
import torch
import warnings
import os
import thop

from ...common.others import get_cur_time_str


class ModelSaveMethod(enum.Enum):
    """
    - WEIGHT: save model by `torch.save(model.state_dict(), ...)`
    - FULL: save model by `torch.save(model, ...)`
    - JIT: convert model to JIT format and save it by `torch.jit.save(jit_model, ...)`
    """
    WEIGHT = 0
    FULL = 1
    JIT = 2
    

def save_model(model: torch.nn.Module,
               model_file_path: str,
               save_method: ModelSaveMethod,
               model_input_size: Tuple[int]=None):
    """Save a PyTorch model.

    Args:
        model (torch.nn.Module): A PyTorch model.
        model_file_path (str): Target model file path.
        save_method (ModelSaveMethod): The method to save model.
        model_input_size (Tuple[int], optional): \
            This is required if :attr:`save_method` is :attr:`ModelSaveMethod.JIT`. \
            Typically be `(1, 3, 32, 32)` or `(1, 3, 224, 224)`. \
            Defaults to None.
    """
    
    model.eval()
    if save_method == ModelSaveMethod.WEIGHT:
        torch.save(model.state_dict(), model_file_path)

    elif save_method == ModelSaveMethod.FULL:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.save(model, model_file_path)

    elif save_method == ModelSaveMethod.JIT:
        assert model_input_size is not None
        dummy_input = torch.ones(model_input_size, device=get_model_device(model))
        new_model = torch.jit.trace(model, dummy_input, check_trace=False)
        torch.jit.save(new_model, model_file_path)
        

def get_model_size(model: torch.nn.Module, return_MB=False):
    """Get size of a PyTorch model (default in Byte).

    Args:
        model (torch.nn.Module): A PyTorch model.
        return_MB (bool, optional): Return result in MB (/= 1024**2). Defaults to False.

    Returns:
        int: Model size.
    """
    pid = os.getpid()
    tmp_model_file_path = './tmp-get-model-size-{}-{}.model'.format(pid, get_cur_time_str())
    save_model(model, tmp_model_file_path, ModelSaveMethod.FULL)

    model_size = os.path.getsize(tmp_model_file_path)
    os.remove(tmp_model_file_path)
    
    if return_MB:
        model_size /= 1024**2

    return model_size


def get_model_device(model: torch.nn.Module):
    """Get device of a PyTorch model.

    Args:
        model (torch.nn.Module): A PyTorch model.

    Returns:
        str: The device of :attr:`model` ('cpu' or 'cuda:x').
    """
    return list(model.parameters())[0].device


def get_model_latency(model: torch.nn.Module, model_input_size: Tuple[int], sample_num: int, 
                      device: str, warmup_sample_num: int, return_detail=False):
    """Get the latency (inference time) of a PyTorch model.
    
    Reference: https://deci.ai/resources/blog/measure-inference-time-deep-neural-networks/

    Args:
        model (torch.nn.Module): A PyTorch model.
        model_input_size (Tuple[int]): Typically be `(1, 3, 32, 32)` or `(1, 3, 224, 224)`.
        sample_num (int): How many inputs which size is :attr:`model_input_size` will be tested and compute the average latency as result.
        device (str): Typically be 'cpu' or 'cuda'.
        warmup_sample_num (int): Let model perform some dummy inference to warm up the test environment to avoid measurement loss.
        return_detail (bool, optional): Beside the average latency, return all result measured. Defaults to False.

    Returns:
        Union[float, Tuple[float, List[float]]]: The average latency (and all lantecy data) of :attr:`model`.
    """
    dummy_input = torch.rand(model_input_size).to(device)
    model = model.to(device)
    model.eval()
    
    # warm up
    with torch.no_grad():
        for _ in range(warmup_sample_num):
            model(dummy_input)
            
    infer_time_list = []
            
    if device == 'cuda':
        with torch.no_grad():
            for _ in range(sample_num):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                model(dummy_input)
                e.record()
                torch.cuda.synchronize()
                cur_model_infer_time = s.elapsed_time(e) / 1000.
                infer_time_list += [cur_model_infer_time]

    else:
        with torch.no_grad():
            for _ in range(sample_num):
                start = time.time()
                model(dummy_input)
                cur_model_infer_time = time.time() - start
                infer_time_list += [cur_model_infer_time]
                
    avg_infer_time = sum(infer_time_list) / sample_num

    if return_detail:
        return avg_infer_time, infer_time_list
    return avg_infer_time


def get_model_flops_and_params(model: torch.nn.Module, model_input_size: Tuple[int]):
    """Get FLOPs and number of parameters of a PyTorch model.

    Args:
        model (torch.nn.Module): A PyTorch model.
        model_input_size (Tuple[int]): Typically be `(1, 3, 32, 32)` or `(1, 3, 224, 224)`.

    Returns:
        Tuple[float, float]: FLOPs and number of parameters of :attr:`model`.
    """
    device = get_model_device(model)
    ops, param = thop.profile(model, (torch.ones(model_input_size).to(device), ), verbose=False)
    return ops * 2, param


def get_module(model: torch.nn.Module, module_name: str):
    """Get a module from a PyTorch model.
    
    Example:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> get_module(model, 'layer1.0')
        BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    Args:
        model (torch.nn.Module): A PyTorch model.
        module_name (str): Module name.

    Returns:
        torch.nn.Module: Corrsponding module.
    """
    for name, module in model.named_modules():
        if name == module_name:
            return module

    return None


def get_super_module(model: torch.nn.Module, module_name: str):
    """Get the super module of a module in a PyTorch model.
    
    Example:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> get_super_module(model, 'layer1.0.conv1')
        BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    Args:
        model (torch.nn.Module): A PyTorch model.
        module_name (str): Module name.

    Returns:
        torch.nn.Module: Super module of module :attr:`module_name`.
    """
    super_module_name = '.'.join(module_name.split('.')[0:-1])
    return get_module(model, super_module_name)


def set_module(model: torch.nn.Module, module_name: str, module: torch.nn.Module):
    """Set module in a PyTorch model.
    
    Example:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> set_module(model, 'layer1.0', torch.nn.Conv2d(64, 64, 3))
        >>> model
        ResNet(
            (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            (layer1): Sequential(
            --> (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (1): BasicBlock(
                    ...
                )
                ...
            )
            ...
        )

    Args:
        model (torch.nn.Module): A PyTorch model.
        module_name (str): Module name.
        module (torch.nn.Module): Target module which will be set into :attr:`model`.
    """
    super_module = get_super_module(model, module_name)
    setattr(super_module, module_name.split('.')[-1], module)


def get_ith_layer(model: torch.nn.Module, i: int):
    """Get i-th layer in a PyTorch model.
    
    Example:
        >>> from torchvision.models import vgg16
        >>> model = vgg16()
        >>> get_ith_layer(model, 5)
        Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    Args:
        model (torch.nn.Module): A PyTorch model.
        i (int): Index of target layer.

    Returns:
        torch.nn.Module: i-th layer in :attr:`model`.
    """
    j = 0
    for module in model.modules():
        if len(list(module.children())) > 0:
            continue
        if j == i:
            return module 
        j += 1
    return None


def get_ith_layer_name(model: torch.nn.Module, i: int):
    """Get the name of i-th layer in a PyTorch model.
    
    Example:
        >>> from torchvision.models import vgg16
        >>> model = vgg16()
        >>> get_ith_layer_name(model, 5)
        'features.5'

    Args:
        model (torch.nn.Module): A PyTorch model.
        i (int): Index of target layer.

    Returns:
        str: The name of i-th layer in :attr:`model`.
    """
    j = 0
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
        if j == i:
            return name
        j += 1
    return None


def set_ith_layer(model: torch.nn.Module, i: int, layer: torch.nn.Module):
    """Set i-th layer in a PyTorch model.
    
    Example:
        >>> from torchvision.models import vgg16
        >>> model = vgg16()
        >>> model
        VGG(
            (features): Sequential(
                (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): ReLU(inplace=True)
                (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                ...
            )
            ...
        )
        >>> set_ith_layer(model, 2, torch.nn.Conv2d(64, 128, 3))
        VGG(
            (features): Sequential(
                (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): ReLU(inplace=True)
            --> (2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                ...
            )
            ...
        )

    Args:
        model (torch.nn.Module): A PyTorch model.
        i (int): Index of target layer.
        layer (torch.nn.Module): The layer which will be set into :attr:`model`.
    """
    j = 0
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
        if j == i:
            set_module(model, name, layer)
            return
        j += 1
        
    
def get_all_specific_type_layers_name(model: torch.nn.Module, types: Tuple[Type[torch.nn.Module]]):
    """Get names of all layers which are give types in a PyTorch model. (e.g. `Conv2d`, `Linear`)
    
    Example:
        >>> from torchvision.models import vgg16
        >>> model = vgg16()
        >>> get_all_specific_type_layers_name(model, (torch.nn.Conv2d))
        ['features.0', 'features.2', 'features.5', ...]

    Args:
        model (torch.nn.Module): A PyTorch model.
        types (Tuple[Type[torch.nn.Module]]): Target types, e.g. `(e.g. torch.nn.Conv2d, torch.nn.Linear)`

    Returns:
        List[str]: Names of all layers which are give types.
    """
    res = []
    for name, m in model.named_modules():
        if isinstance(m, types):
            res += [name]
    return res


class LayerActivation:
    """Collect the input and output of a middle module of a PyTorch model during inference.
    
    Layer is a wide concept in this class. A module (e.g. ResBlock in ResNet) can be also regarded as a "layer".
    
    Example:
        >>> from torchvision.models import vgg16
        >>> model = vgg16()
        >>> # collect the input and output of 5th layer in VGG16
        >>> layer_activation = LayerActivation(get_ith_layer(model, 5), 'cuda')
        >>> model(torch.rand((1, 3, 224, 224)))
        >>> layer_activation.input
        tensor([[...]])
        >>> layer_activation.output
        tensor([[...]])
        >>> layer_activation.remove()
    """
    def __init__(self, layer: torch.nn.Module, device: str):
        """Register forward hook on corresponding layer.

        Args:
            layer (torch.nn.Module): Target layer.
            device (str): Where the collected data is located.
        """
        self.hook = layer.register_forward_hook(self._hook_fn)
        self.device = device
        self.input: torch.Tensor = None
        self.output: torch.Tensor = None
        self.layer = layer
        
    def __str__(self):
        return '- ' + str(self.layer)

    def _hook_fn(self, module, input, output):
        # TODO: input or output may be a tuple
        if isinstance(input, tuple):
            self.input = input[0].detach().to(self.device)
        else:
            self.input = input.detach().to(self.device)

        if isinstance(output, tuple):
            self.output = output[0].detach().to(self.device)
        else:
            self.output = output.detach().to(self.device)

    def remove(self):
        """Remove the hook in the model to avoid performance effect.
        Use this after using the collected data.
        """
        self.hook.remove()


class LayerActivationWrapper:
    """A wrapper of :attr:`LayerActivation` which has the same API, but broaden the concept "layer".
    Now a series of layers can be regarded as "hyper-layer" in this class.
    
    Example:
        >>> from torchvision.models import vgg16
        >>> model = vgg16()
        >>> # collect the input of 5th layer, and output of 7th layer in VGG16
        >>> # i.e. regard 5th~7th layer as a whole module, 
        >>> # and collect the input and output of this module
        >>> layer_activation = LayerActivationWrapper([
            LayerActivation(get_ith_layer(model, 5), 'cuda'),
            LayerActivation(get_ith_layer(model, 6), 'cuda')
            LayerActivation(get_ith_layer(model, 7), 'cuda')
        ])
        >>> model(torch.rand((1, 3, 224, 224)))
        >>> layer_activation.input
        tensor([[...]])
        >>> layer_activation.output
        tensor([[...]])
        >>> layer_activation.remove()
    """
    def __init__(self, las: List[LayerActivation]):
        """
        Args:
            las (List[LayerActivation]): The layer activations of a series of layers.
        """
        self.las = las
        
    def __str__(self):
        return '\n'.join([str(la) for la in self.las])

    @property
    def input(self):
        """Get the collected input data of first layer.

        Returns:
            torch.Tensor: Collected input data of first layer.
        """
        return self.las[0].input

    @property
    def output(self):
        """Get the collected input data of last layer.

        Returns:
            torch.Tensor: Collected input data of last layer.
        """
        return self.las[-1].output

    def remove(self):
        """Remove all hooks in the model to avoid performance effect.
        Use this after using the collected data.
        """
        [la.remove() for la in self.las]


class TimeProfiler:
    """ (NOT VERIFIED. DON'T USE ME)
    """
    def __init__(self, layer: torch.nn, device):
        self.before_infer_hook = layer.register_forward_pre_hook(self.before_hook_fn)
        self.after_infer_hook = layer.register_forward_hook(self.after_hook_fn)

        self.device = device
        self.infer_time = None
        self._start_time = None
        
        if self.device != 'cpu':
            self.s, self.e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def before_hook_fn(self, module, input):
        if self.device == 'cpu':
            self._start_time = time.time()
        else:
            self.s.record()

    def after_hook_fn(self, module, input, output):
        if self.device == 'cpu':
            self.infer_time = time.time() - self._start_time
        else:
            self.e.record()
            torch.cuda.synchronize()
            self.infer_time = self.s.elapsed_time(self.e) / 1000.

    def remove(self):
        self.before_infer_hook.remove()
        self.after_infer_hook.remove()


class TimeProfilerWrapper:
    """ (NOT VERIFIED. DON'T USE ME)
    """
    def __init__(self, tps: List[TimeProfiler]):
        self.tps = tps

    @property
    def infer_time(self):
        return sum([tp.infer_time for tp in self.tps])

    def remove(self):
        [tp.remove() for tp in self.tps]