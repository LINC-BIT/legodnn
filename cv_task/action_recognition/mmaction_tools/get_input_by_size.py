import warnings

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
import os.path as osp
import os
import re
from mmaction.datasets.pipelines import Compose
from operator import itemgetter
from mmaction.core import OutputHook

def get_input_by_size(model, input_size):
    video = 'cv_task/action_recognition/mmaction_tools/demo.mp4'

    input_flag = None
    if isinstance(video, dict):
        input_flag = 'dict'
    elif isinstance(video, np.ndarray):
        assert len(video.shape) == 4, 'The shape should be T x H x W x C'
        input_flag = 'array'
    elif isinstance(video, str) and video.startswith('http'):
        input_flag = 'video'
    elif isinstance(video, str) and osp.exists(video):
        if osp.isfile(video):
            input_flag = 'video'
        if osp.isdir(video):
            input_flag = 'rawframes'
    else:
        raise RuntimeError('The type of argument video is not supported: '
                           f'{type(video)}')

    # if isinstance(outputs, str):
    #     outputs = (outputs, )
    # assert outputs is None or isinstance(outputs, (tuple, list))

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline
    # Alter data pipelines & prepare inputs
    if input_flag == 'dict':
        data = video
    if input_flag == 'array':
        modality_map = {2: 'Flow', 3: 'RGB'}
        modality = modality_map.get(video.shape[-1])
        data = dict(
            total_frames=video.shape[0],
            label=-1,
            start_index=0,
            array=video,
            modality=modality)
        for i in range(len(test_pipeline)):
            if 'Decode' in test_pipeline[i]['type']:
                test_pipeline[i] = dict(type='ArrayDecode')
    if input_flag == 'video':
        data = dict(filename=video, label=-1, start_index=0, modality='RGB')
        if 'Init' not in test_pipeline[0]['type']:
            test_pipeline = [dict(type='OpenCVInit')] + test_pipeline
        else:
            test_pipeline[0] = dict(type='OpenCVInit')
        for i in range(len(test_pipeline)):
            if 'Decode' in test_pipeline[i]['type']:
                test_pipeline[i] = dict(type='OpenCVDecode')
    if input_flag == 'rawframes':
        filename_tmpl = cfg.data.test.get('filename_tmpl', 'img_{:05}.jpg')
        modality = cfg.data.test.get('modality', 'RGB')
        start_index = cfg.data.test.get('start_index', 1)

        # count the number of frames that match the format of `filename_tmpl`
        # RGB pattern example: img_{:05}.jpg -> ^img_\d+.jpg$
        # Flow patteren example: {}_{:05d}.jpg -> ^x_\d+.jpg$
        pattern = f'^{filename_tmpl}$'
        if modality == 'Flow':
            pattern = pattern.replace('{}', 'x')
        pattern = pattern.replace(
            pattern[pattern.find('{'):pattern.find('}') + 1], '\\d+')
        total_frames = len(
            list(
                filter(lambda x: re.match(pattern, x) is not None,
                       os.listdir(video))))
        data = dict(
            frame_dir=video,
            total_frames=total_frames,
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
        if 'Init' in test_pipeline[0]['type']:
            test_pipeline = test_pipeline[1:]
        for i in range(len(test_pipeline)):
            if 'Decode' in test_pipeline[i]['type']:
                test_pipeline[i] = dict(type='RawFrameDecode')

    test_pipeline = Compose(test_pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # # forward the model
    # with OutputHook(model, outputs=outputs, as_tensor=as_tensor) as h:
    #     with torch.no_grad():
    #         scores = model(return_loss=False, **data)[0]
    #     returned_features = h.layer_outputs if outputs else None

    # num_classes = scores.shape[-1]
    # score_tuples = tuple(zip(range(num_classes), scores))
    # score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)

    # top5_label = score_sorted[:5]
    # if outputs:
    #     return top5_label, returned_features
    # return top5_label
    return data
# if __name__=='__main__':
#     print(aaa
#     tensor_to_datacontainer