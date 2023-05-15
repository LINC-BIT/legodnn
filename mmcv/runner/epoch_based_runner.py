# Copyright (c) OpenMMLab. All rights reserved.
from locale import NOEXPR
import os.path as osp
import platform
import shutil
import time
import warnings
import random
import torch
import copy
import sys

import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info

# sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/fn3')
# from fn3_channel_open_api_1215.fn3_channel import set_fn3_channel_channels, export_active_sub_net

# sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/cgnet')
# from cgnet_open_api_1212 import convert_model_to_cgnet, add_cgnet_loss, get_cgnet_flops_save_ratio, get_cgnet_flops_save_ratio

from legodnn.utils.dl.common.model import get_module
# from baselines.nested_network.nestdnn_1230.nestdnn_open_api import zero_grads_nestdnn_layers, freeze_no_nestdnn_layers

@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch') # call hook用来干什么？？？
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        self.method = kwargs.get('method', None)
        original_kwargs = copy.deepcopy(kwargs)
        # remove args ralated to usnet 
        if self.method == 'usnet':
            original_kwargs.pop('method')
            width_mult_list = kwargs.get('width_mult_list', None)
            sample_net_num = kwargs.get('sample_net_num', None)   
            original_kwargs.pop('width_mult_list')
            original_kwargs.pop('sample_net_num')
        
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            if self.method is not None:
                
                if self.method == 'usnet':
                    assert width_mult_list is not None
                    min_width, max_width = min(width_mult_list), max(width_mult_list)
                    widths_train = []
                    
                    for _ in range(sample_net_num - 2):
                        widths_train.append(
                            random.uniform(min_width, max_width))
                    widths_train = [max_width, min_width] + widths_train
                
                    self.call_hook('before_train_iter')  # zero grad
                    for width in widths_train:
                        # print(width)
                        self.model.apply(lambda m: setattr(m, 'width_mult', width))
                        self.run_iter(data_batch, train_mode=True, **original_kwargs)  # forward usnet in every width and compute loss
                        # print(self.outputs['loss'])
                        self.outputs['loss'].backward()  # backward
                    self.call_hook('after_train_iter')  # step
                    self.model.apply(lambda m: setattr(m, 'width_mult', 1.0))
                    
                elif self.method=='fn3':
                    fn3_all_layers = original_kwargs.get('fn3_all_layers', None)
                    fn3_disable_layers = original_kwargs.get('fn3_disable_layers', None)
                    min_sparsity = original_kwargs.get('min_sparsity', None)
                    
                    assert fn3_all_layers is not None
                    assert fn3_disable_layers is not None
                    
                    fn3_channel_layers_name = [i[0] for i in fn3_all_layers]
                    fn3_channel_channels = [i[1] for i in fn3_all_layers]
                    # print(fn3_channel_channels)
                    # print(type(fn3_channel_channels[0]))
                    for i, c in enumerate(fn3_channel_channels):
                        if fn3_channel_layers_name[i] in fn3_disable_layers:
                            continue
                        fn3_channel_channels[i] = random.randint(int(min_sparsity*c), c)
                        
                    set_fn3_channel_channels(self.model, fn3_channel_channels)
                    
                    self.call_hook('before_train_iter')  # zero grad
                    self.run_iter(data_batch, train_mode=True, **original_kwargs)  # forward fn3
                    self.call_hook('after_train_iter')  # compute loss, step

                    set_fn3_channel_channels(self.model, [i[1] for i in fn3_all_layers])
                    
                elif self.method=='nestdnn':
                    grad_positions = original_kwargs.get('grad_positions', None)
                    assert grad_positions is not None
                    
                    self.call_hook('before_train_iter') # zero grad
                    self.run_iter(data_batch, train_mode=True, **original_kwargs)  # forward nestdnn
                    self.outputs['loss'].backward()  # compute loss
                    
                    # for name, param in self.model.module.named_parameters():
                    #     print(name)
                    #     print(param.requires_grad)
                    
                    for key, values in grad_positions.items():
                        # print(key)
                        # print(self.model.module)
                        nest_module = get_module(self.model.module, key)
                        # print(nest_module)
                        for value_key, value in values.items():
                            if value_key.startswith('weight'):
                                # print(nest_module.weight)
                                # print(nest_module.weight.requires_grad)
                                # print(nest_module.weight.grad)
                                assert len(value)==1 or len(value)==4
                                if len(value)==1:
                                    nest_module.weight.grad[:value[0]].data.zero_()
                                elif len(value)==4:
                                    nest_module.weight.grad[:value[0], :value[1], :value[2], :value[3]].data.zero_()
                                else:
                                    raise NotImplementedError
                                # print(nest_module.weight)
                                # print(nest_module.weight.requires_grad)
                                # print(nest_module.weight.grad)
                                
                            elif value_key.startswith('bias'):
                                assert len(value)==1
                                nest_module.bias.grad[:value[0]].data.zero_()
                            # elif value_key.startswith('running_mean'):
                            #     assert len(value)==0
                            #     nest_module.running_mean.grad[:value[0]].data = 0.0
                            # elif value_key.startswith('running_var'):
                            #     nest_module.running_var.grad[:value[0]].data = 0.0
                            else:
                                raise NotImplementedError
                            # exit(0)
                        pass
                    
                    self.call_hook('after_train_iter')  # step
                elif self.method=='nestdnnv3':
                    freeze_no_nestdnn_layers(self.model.module)
                    zero_shape_info = original_kwargs.get('zero_shape_info', None)
                    assert zero_shape_info is not None
                    
                    self.call_hook('before_train_iter') # zero grad
                    self.run_iter(data_batch, train_mode=True, **original_kwargs)  # forward nestdnn
                    self.outputs['loss'].backward()  # compute loss
                    
                    zero_grads_nestdnn_layers(self.model.module, zero_shape_info) # 清空前一个模型的梯度
                    
                    self.call_hook('after_train_iter')  # step
            else:
                self.call_hook('before_train_iter')
                self.run_iter(data_batch, train_mode=True, **kwargs)
                self.call_hook('after_train_iter')
            
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


@RUNNERS.register_module()
class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead')
        super().__init__(*args, **kwargs)
