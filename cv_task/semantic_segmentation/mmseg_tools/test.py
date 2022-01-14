# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

from torch.cuda import device

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor



def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('-config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args




def test_segmentor(segmentor, test_loader):
    args = parse_args()

    assert segmentor.cfg.get('evaluation') is not None
    
    args.eval = segmentor.cfg.evaluation.metric
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    # cfg = mmcv.Config.fromfile(args.config)
    cfg = segmentor.cfg
    
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    # if cfg.get('cudnn_benchmark', False):
    #     torch.backends.cudnn.benchmark = True
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # if args.aug_test:
    #     # hard code index
    #     cfg.data.test.pipeline[1].img_ratios = [
    #         0.5, 0.75, 1.0, 1.25, 1.5, 1.75
    #     ]
    #     cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    # if args.launcher == 'none':
    #     distributed = False
    # else:
    #     distributed = True
    #     init_dist(args.launcher, **cfg.dist_params)
    distributed = False
    
    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    # dataset = build_dataset(cfg.data.test)
    # data_loader = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     dist=distributed,
    #     shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    # model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    model = segmentor

    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # if 'CLASSES' in checkpoint.get('meta', {}):
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
    #     print('"CLASSES" not found in meta, use dataset.CLASSES instead')
    #     model.CLASSES = test_loader.dataset.CLASSES
    # if 'PALETTE' in checkpoint.get('meta', {}):
    #     model.PALETTE = checkpoint['meta']['PALETTE']
    # else:
    #     print('"PALETTE" not found in meta, use dataset.PALETTE instead')
    #     model.PALETTE = test_loader.dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options

    # Deprecated
    # efficient_test = eval_kwargs.get('efficient_test', False)
    # if efficient_test:
    #     warnings.warn(
    #         '``efficient_test=True`` does not have effect in tools/test.py, '
    #         'the evaluation and format results are CPU memory efficient by '
    #         'default')

    # eval_on_format_results = (
    #     args.eval is not None and 'cityscapes' in args.eval)
    # if eval_on_format_results:
    #     assert len(args.eval) == 1, 'eval on format results is not ' \
    #                                 'applicable for metrics other than ' \
    #                                 'cityscapes'
    # if args.format_only or eval_on_format_results:
    #     if 'imgfile_prefix' in eval_kwargs:
    #         tmpdir = eval_kwargs['imgfile_prefix']
    #     else:
    #         tmpdir = '.format_cityscapes'
    #         eval_kwargs.setdefault('imgfile_prefix', tmpdir)
    #     mmcv.mkdir_or_exist(tmpdir)
    # else:
    #     tmpdir = None

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        results = single_gpu_test(
            model,
            test_loader,
            args.show,
            args.show_dir,
            False,
            args.opacity)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        results = multi_gpu_test(
            model,
            test_loader,
            args.tmpdir,
            args.gpu_collect,
            False)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            warnings.warn(
                'The behavior of ``args.out`` has been changed since MMSeg '
                'v0.16, the pickled outputs could be seg map as type of '
                'np.array, pre-eval results or file paths for '
                '``dataset.format_results()``.')
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)
        if args.eval:
            eval_kwargs.update(metric=args.eval)
            metric = test_loader.dataset.evaluate(results, **eval_kwargs)
            # print(11111)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file, indent=4)
    
    print('评测结果：')
    print(metric[args.eval])
    return metric[args.eval]

if __name__ == '__main__':

    from cv_task.semantic_segmentation.mmseg_models.legodnn_configs.__init__ import get_fcn_r50_d8_512x1024_80k_cityscapes_config
    from cv_task.semantic_segmentation.mmseg_models.fcn import fcn_r50_d8
    from cv_task.datasets.semantic_segmentation.mmseg_cityscapes import mmseg_cityscapes_dataloader
    
    model_input_size = (1, 3, 320, 320)
    device = 'cuda'
    model_config = get_fcn_r50_d8_512x1024_80k_cityscapes_config(input_size=model_input_size)
    segmentor = fcn_r50_d8(config=model_config, mode='mmseg_test', device=device)
    train_loader, test_loader = mmseg_cityscapes_dataloader(cfg=model_config)
    
    map = test_segmentor(segmentor, test_loader)
    print(map)