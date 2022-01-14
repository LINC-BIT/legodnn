# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model
from torch.utils.data import distributed

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import register_module_hooks

# TODO import test functions from mmcv and delete them from mmaction2
try:
    from mmcv.engine import multi_gpu_test, single_gpu_test
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'DeprecationWarning: single_gpu_test, multi_gpu_test, '
        'collect_results_cpu, collect_results_gpu from mmaction2 will be '
        'deprecated. Please install mmcv through master branch.')
    from mmaction.apis import multi_gpu_test, single_gpu_test


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--onnx',
        action='store_true',
        help='Whether to test with onnx model or not')
    parser.add_argument(
        '--tensorrt',
        action='store_true',
        help='Whether to test with TensorRT engine or not')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


def inference_pytorch(recognizer, args, cfg, distributed, data_loader):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    # build the model and load checkpoint
    # model = build_model(
    #     cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    model = recognizer
    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    # load_checkpoint(model, args.checkpoint, map_location='cpu')

    # if args.fuse_conv_bn:
    #     model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    return outputs


# recognizer = init_recognizer(config, checkpoint, device=device)
def test_recognizer(recognizer, test_loader,  eval='top_k_accuracy', dataset_name='hmdb51'):
    args = parse_args()
    args.eval = eval
    if args.tensorrt and args.onnx:
        raise ValueError(
            'Cannot set onnx mode and tensorrt mode at the same time.')

    cfg = recognizer.cfg

    # cfg.merge_from_dict(args.cfg_options)

    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    if args.out:
        # Overwrite output_config from args.out
        output_config = Config._merge_a_into_b(
            dict(out=args.out), output_config)

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    if args.eval:
        # Overwrite eval_config from args.eval
        eval_config = Config._merge_a_into_b(
            dict(metrics=args.eval), eval_config)
    if args.eval_options:
        # Add options from args.eval_options
        eval_config = Config._merge_a_into_b(args.eval_options, eval_config)

    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" or "--eval"')

    dataset_type = cfg.data.test.type
    if output_config.get('out', None):
        if 'output_format' in output_config:
            # ugly workround to make recognition and localization the same
            warnings.warn(
                'Skip checking `output_format` in localization task.')
        else:
            out = output_config['out']
            # make sure the dirname of the output path exists
            mmcv.mkdir_or_exist(osp.dirname(out))
            _, suffix = osp.splitext(out)
            if dataset_type == 'AVADataset':
                assert suffix[1:] == 'csv', ('For AVADataset, the format of '
                                             'the output file should be csv')
            else:
                assert suffix[1:] in file_handlers, (
                    'The format of the output '
                    'file should be json, pickle or yaml')

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    # if args.launcher == 'none':
    #     distributed = False
    # else:
    #     distributed = True
    #     init_dist(args.launcher, **cfg.dist_params)
    distributed = False
    
    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # build the dataloader
    # dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    # dataloader_setting = dict(
    #     videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
    #     workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
    #     dist=distributed,
    #     shuffle=False)
    # dataloader_setting = dict(dataloader_setting,
    #                           **cfg.data.get('test_dataloader', {}))
    # data_loader = build_dataloader(dataset, **dataloader_setting)

    # if args.tensorrt:
    #     outputs = inference_tensorrt(args.checkpoint, distributed, data_loader,
    #                                  dataloader_setting['videos_per_gpu'])
    # elif args.onnx:
    #     outputs = inference_onnx(args.checkpoint, distributed, data_loader,
    #                              dataloader_setting['videos_per_gpu'])
    # else:
    #     outputs = inference_pytorch(args, cfg, distributed, data_loader)
    outputs = inference_pytorch(recognizer, args, cfg, distributed, test_loader)
    
    rank, _ = get_dist_info()
    if rank == 0:
        if output_config.get('out', None):
            out = output_config['out']
            print(f'\nwriting results to {out}')
            test_loader.dataset.dump_results(outputs, **output_config)
        if eval_config:
            eval_res = test_loader.dataset.evaluate(outputs, **eval_config)
            for name, val in eval_res.items():
                print(f'{name}: {val:.04f}')
    return eval_res['top1_acc']

if __name__ == '__main__':
    from cv_task.action_recognition.mmaction_models.legodnn_configs import get_tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb_config
    from cv_task.action_recognition.mmaction_models.tsn import tsn_r50_1x1x8_50e_hmdb51_imagenet
    from cv_task.datasets.action_recognition.mmaction_hmdb51 import mmaction_hmdb51_dataloader
    
    model_config = get_tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb_config()
    # print(model_config)
    # exit(0)
    model = tsn_r50_1x1x8_50e_hmdb51_imagenet(model_config, mode='mmaction_test')
    train_loader, test_loader = mmaction_hmdb51_dataloader(model_config, 64, 8, 1)
    print(len(test_loader))
    # exit(0)
    # exit(0)
    acc = test_recognizer(model, test_loader)
    print(acc)
