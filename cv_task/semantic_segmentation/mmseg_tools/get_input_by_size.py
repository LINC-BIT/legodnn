import warnings

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.apis.inference import LoadImage, Compose

def get_input_by_size(model, input_size):
    img = '/data/gxy/legodnn-public-version_semantic_segmentation/cv_task/semantic_segmentation/mmseg_tools/demo.png'

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    return data

# if __name__=='__main__':
#     print(aaa
#     tensor_to_datacontainer