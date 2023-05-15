# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, EpochBasedRunner, OptimizerHook

from mmpose.core import DistEvalHook, EvalHook, build_optimizers
from mmpose.core.distributed_wrapper import DistributedDataParallelWrapper
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.utils import get_root_logger

try:
    from mmcv.runner import Fp16OptimizerHook
except ImportError:
    warnings.warn(
        'Fp16OptimizerHook from mmpose will be deprecated from '
        'v0.15.0. Please install mmcv>=1.1.4', DeprecationWarning)
    from mmpose.core import Fp16OptimizerHook


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (Dataset): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(
            seed=cfg.get('seed'),
            drop_last=False,
            dist=distributed,
            num_gpus=len(cfg.gpu_ids)),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'samples_per_gpu',
                   'workers_per_gpu',
                   'shuffle',
                   'seed',
                   'drop_last',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }

    # step 2: cfg.data.train_dataloader has highest priority
    train_loader_cfg = dict(loader_cfg, **cfg.data.get('train_dataloader', {}))

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # determine whether use adversarial training precess or not
    use_adverserial_train = cfg.get('use_adversarial_train', False)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel

        if use_adverserial_train:
            # Use DistributedDataParallelWrapper for adversarial training
            model = DistributedDataParallelWrapper(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizers(model, cfg.optimizer)

    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    if use_adverserial_train:
        # The optimizer step process is included in the train_step function
        # of the model, so the runner should NOT include optimizer hook.
        optimizer_config = None
    else:
        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
        elif distributed and 'type' not in cfg.optimizer_config:
            optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            drop_last=False,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    method = cfg.get('method', None)
    if method is not None:
        if method=='usnet':
            width_mult_list = cfg.get('width_mult_list', None)
            sample_net_num = cfg.get('sample_net_num', None)
            assert width_mult_list is not None
            assert sample_net_num is not None
            runner.run(data_loaders, cfg.workflow, cfg.total_epochs, method=method, width_mult_list=width_mult_list, sample_net_num=sample_net_num)
            # cal
            # runner.run(data_loaders, cfg.workflow, max_iters=len(data_loaders), method='usnet_cal', width_mult_list=width_mult_list, sample_net_num=sample_net_num)
        elif method=='fn3':
            fn3_all_layers = cfg.get('fn3_all_layers', None)
            fn3_disable_layers = cfg.get('fn3_disable_layers', None)
            min_sparsity = cfg.get('min_sparsity', None)
            
            assert fn3_all_layers is not None
            assert fn3_disable_layers is not None
            runner.run(data_loaders, cfg.workflow, cfg.total_epochs, method=method, fn3_all_layers=fn3_all_layers, fn3_disable_layers=fn3_disable_layers, min_sparsity=min_sparsity)
            
        elif method=='cgnet':
            gtar = cfg.get('gtar', None)
            input_size = cfg.get('input_size', None)
            assert gtar is not None and input_size is not None
            runner.run(data_loaders, cfg.workflow, cfg.total_epochs, method=method, gtar=gtar, input_size=input_size)
        elif method=='nestdnn':
            grad_positions = cfg.get('grad_positions', None)
            # freeze = cfg.get('freeze', None)
            assert grad_positions is not None
            runner.run(data_loaders, cfg.workflow, cfg.total_epochs, method=method, grad_positions=grad_positions)
        elif method=='nestdnnv3':
            zero_shape_info = cfg.get('zero_shape_info', None)
            # freeze = cfg.get('freeze', None)
            assert zero_shape_info is not None
            runner.run(data_loaders, cfg.workflow, cfg.total_epochs, method=method, zero_shape_info=zero_shape_info)
        else:
            raise NotImplementedError
    else:
        runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
    
