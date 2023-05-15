# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner

from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    # runner.run(data_loaders, cfg.workflow)
    method = cfg.get('method', None)
    if method is not None:
        print("method name: {}".format(method))
        if method=='usnet':
            width_mult_list = cfg.get('width_mult_list', None)
            sample_net_num = cfg.get('sample_net_num', None)
            assert width_mult_list is not None
            assert sample_net_num is not None
            runner.run(data_loaders, cfg.workflow, method=method, width_mult_list=width_mult_list, sample_net_num=sample_net_num)
            # cal
            # runner.run(data_loaders, cfg.workflow, max_iters=len(data_loaders), method='usnet_cal', width_mult_list=width_mult_list, sample_net_num=sample_net_num)
        elif method=='fn3':
            fn3_all_layers = cfg.get('fn3_all_layers', None)
            fn3_disable_layers = cfg.get('fn3_disable_layers', None)
            min_sparsity = cfg.get('min_sparsity', None)
            
            assert fn3_all_layers is not None
            assert fn3_disable_layers is not None
            runner.run(data_loaders, cfg.workflow, method=method, fn3_all_layers=fn3_all_layers, fn3_disable_layers=fn3_disable_layers, min_sparsity=min_sparsity)
            
        elif method=='cgnet':
            gtar = cfg.get('gtar', None)
            input_size = cfg.get('input_size', None)
            assert gtar is not None and input_size is not None
            runner.run(data_loaders, cfg.workflow, method=method, gtar=gtar, input_size=input_size)
        elif method=='nestdnn':
            grad_positions = cfg.get('grad_positions', None)
            # freeze = cfg.get('freeze', None)
            assert grad_positions is not None
            runner.run(data_loaders, cfg.workflow, method=method, grad_positions=grad_positions)
        elif method=='nestdnnv3':
            zero_shape_info = cfg.get('zero_shape_info', None)
            assert zero_shape_info is not None
            runner.run(data_loaders, cfg.workflow, method=method, zero_shape_info=zero_shape_info)
        else:
            raise NotImplementedError
    else:
        runner.run(data_loaders, cfg.workflow)
