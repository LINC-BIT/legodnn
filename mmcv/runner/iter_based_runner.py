# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
import copy
import random

import torch
from torch.optim import Optimizer

import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .hooks import IterTimerHook
from .utils import get_host_info
import sys
from legodnn.utils.dl.common.model import make_divisible
# sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/slimmable_networks')
# from usnet_open_api_1215.us_net import convert_model_to_us_net, set_us_net_width_mult, export_jit_us_sub_net, bn_calibration_init, set_FLAGS

# from baselines.nested_network.nestdnn_1230.nestdnn_open_api import zero_grads_nestdnn_layers

# sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/fn3')
# from fn3_channel_open_api_1215.fn3_channel import set_fn3_channel_channels, export_active_sub_net

# sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/cgnet')
# from cgnet_open_api_1212 import convert_model_to_cgnet, add_cgnet_loss, get_cgnet_flops_save_ratio, get_cgnet_flops_save_ratio

from legodnn.utils.dl.common.model import get_module

class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


@RUNNERS.register_module()
class IterBasedRunner(BaseRunner):
    """Iteration-based Runner.

    This runner train models iteration by iteration.
    """

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.method = kwargs.get('method', None)
        original_kwargs = copy.deepcopy(kwargs)
        # remove args ralated to usnet 
        if self.method == 'usnet':
            original_kwargs.pop('method')
            width_mult_list = kwargs.get('width_mult_list', None)
            sample_net_num = kwargs.get('sample_net_num', None)   
            original_kwargs.pop('width_mult_list')
            original_kwargs.pop('sample_net_num')
            
        
        if self.method is not None:  
            if self.method == 'usnet':
                # cal = kwargs.get('cal', None)
                # if cal is not None and cal:
                #     outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
                #     if not isinstance(outputs, dict):
                #         raise TypeError('model.train_step() must return a dict')
                #     if 'log_vars' in outputs:
                #         self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
                #     self.outputs = outputs
                # else:
                assert width_mult_list is not None
                min_width, max_width = min(width_mult_list), max(width_mult_list)
                widths_train = []
                for _ in range(sample_net_num - 2):
                    widths_train.append(
                        random.uniform(min_width, max_width))
                widths_train = [max_width, min_width] + widths_train
                
                # zero grad
                self.call_hook('before_train_iter')  
                for width in widths_train:
                    self.model.apply(lambda m: setattr(m, 'width_mult', width))
                    # forward usnet in every width and compute loss
                    outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
                    if not isinstance(outputs, dict):
                        raise TypeError('model.train_step() must return a dict')
                    if 'log_vars' in outputs:
                        self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
                    self.outputs = outputs
                    # backward
                    self.outputs['loss'].backward() 
                # step 
                self.call_hook('after_train_iter')  
                self.model.apply(lambda m: setattr(m, 'width_mult', 1.0))
            
            elif self.method=='fn3':
                # print(original_kwargs)
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
                
                self.call_hook('before_train_iter')
                outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
                if not isinstance(outputs, dict):
                    raise TypeError('model.train_step() must return a dict')
                if 'log_vars' in outputs:
                    self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
                self.outputs = outputs
                self.call_hook('after_train_iter')

                set_fn3_channel_channels(self.model, [i[1] for i in fn3_all_layers])
                # pass
            elif self.method=='cgnet':
                gtar = original_kwargs.get('gtar', None)
                input_size = original_kwargs.get('input_size', None)
                assert gtar is not None and input_size is not None
                self.call_hook('before_train_iter')
                
                outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
                if not isinstance(outputs, dict):
                    raise TypeError('model.train_step() must return a dict')
                if 'log_vars' in outputs:
                    self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
                self.outputs = outputs
                add_cgnet_loss(self.model, self.outputs['loss'], gtar)
                self.call_hook('after_train_iter')
                if self.iter%1000==0:
                    print("cgnet flops drop: {}".format(get_cgnet_flops_save_ratio(self.model.module, torch.randn(input_size).cuda())))
                
            elif self.method=='nestdnn':
                grad_positions = original_kwargs.get('grad_positions', None)
                assert grad_positions is not None
                
                self.call_hook('before_train_iter')
                outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
                if not isinstance(outputs, dict):
                    raise TypeError('model.train_step() must return a dict')
                if 'log_vars' in outputs:
                    self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
                self.outputs = outputs
                self.outputs['loss'].backward()
                
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
                
                self.call_hook('after_train_iter')
            elif self.method=='nestdnnv3':
                zero_shape_info = original_kwargs.get('zero_shape_info', None)
                assert zero_shape_info is not None
                
                self.call_hook('before_train_iter')
                outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
                if not isinstance(outputs, dict):
                    raise TypeError('model.train_step() must return a dict')
                if 'log_vars' in outputs:
                    self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
                self.outputs = outputs
                self.outputs['loss'].backward()
                
                zero_grads_nestdnn_layers(self.model.module, zero_shape_info) # 清空前一个模型的梯度
                
                self.call_hook('after_train_iter')
            else:
                raise NotImplementedError
        else:
            self.call_hook('before_train_iter')
            outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('model.train_step() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_train_iter')
        
        self._inner_iter += 1
        self._iter += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        
        # if self.method=='usnet':
        #     # pass
        #     # 第二步： bn init
        #     # 第一步： self.model.apply(lambda m: setattr(m, 'width_mult', 1.0))
        #     bn_calibration_init(self.model)
        #     kwargs['cal'] == True
        #     with torch.no_grad():
        #         self.train()
        #     # 第三步： self.train(cal)
            
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        data_batch = next(data_loader)
        self.call_hook('before_val_iter')
        outputs = self.model.val_step(data_batch, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.val_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_val_iter')
        self._inner_iter += 1

    def run(self, data_loaders, workflow, max_iters=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        iter_loaders = [IterLoader(x) for x in data_loaders]

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        """Resume model from checkpoint.

        Args:
            checkpoint (str): Checkpoint to resume from.
            resume_optimizer (bool, optional): Whether resume the optimizer(s)
                if the checkpoint file includes optimizer(s). Default to True.
            map_location (str, optional): Same as :func:`torch.load`.
                Default to 'default'.
        """
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        self._inner_iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        self.logger.info(f'resumed from epoch: {self.epoch}, iter {self.iter}')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='iter_{}.pth',
                        meta=None,
                        save_optimizer=True,
                        create_symlink=True):
        """Save checkpoint to file.

        Args:
            out_dir (str): Directory to save checkpoint files.
            filename_tmpl (str, optional): Checkpoint file template.
                Defaults to 'iter_{}.pth'.
            meta (dict, optional): Metadata to be saved in checkpoint.
                Defaults to None.
            save_optimizer (bool, optional): Whether save optimizer.
                Defaults to True.
            create_symlink (bool, optional): Whether create symlink to the
                latest checkpoint file. Defaults to True.
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

        filename = filename_tmpl.format(self.iter + 1)
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

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                momentum_config=None,
                                custom_hooks_config=None):
        """Register default hooks for iter-based training.

        Checkpoint hook, optimizer stepper hook and logger hooks will be set to
        `by_epoch=False` by default.

        Default hooks include:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | LrUpdaterHook        | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | MomentumUpdaterHook  | HIGH (30)               |
        +----------------------+-------------------------+
        | OptimizerStepperHook | ABOVE_NORMAL (40)       |
        +----------------------+-------------------------+
        | CheckpointSaverHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | IterTimerHook        | LOW (70)                |
        +----------------------+-------------------------+
        | LoggerHook(s)        | VERY_LOW (90)           |
        +----------------------+-------------------------+
        | CustomHook(s)        | defaults to NORMAL (50) |
        +----------------------+-------------------------+

        If custom hooks have same priority with default hooks, custom hooks
        will be triggered after default hooks.
        """
        if checkpoint_config is not None:
            checkpoint_config.setdefault('by_epoch', False)
        if lr_config is not None:
            lr_config.setdefault('by_epoch', False)
        if log_config is not None:
            for info in log_config['hooks']:
                info.setdefault('by_epoch', False)
        super(IterBasedRunner, self).register_training_hooks(
            lr_config=lr_config,
            momentum_config=momentum_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
            log_config=log_config,
            timer_config=IterTimerHook(),
            custom_hooks_config=custom_hooks_config)
