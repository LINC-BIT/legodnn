# Copyright 2018-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import dlutils
import random
import pickle
# from defaults import get_cfg_defaults
import numpy as np
from os import path
from scipy import misc
# import logging
# from utils.log import logger
from PIL import Image


def get_all_samples(dataset, isize):
    if dataset == 'mnist':
        mnist = dlutils.reader.Mnist('/data/zql/datasets/MNIST/raw', train=True, test=True).items

        images = [x[1] for x in mnist]
        labels = [x[0] for x in mnist]
        images = np.asarray(images)

        assert(images.shape == (70000, 28, 28))

        _images = []
        for im in images:
            # im = misc.imresize(im, (isize, isize), interp='bilinear')
            im = np.array(Image.fromarray(im).resize((isize, isize)))
            _images.append(im)
        images = np.asarray(_images)

        assert(images.shape == (70000, isize, isize))

    return [(im, l) for l, im in zip(labels, images)]


def partition(dataset, fold_num, isize):
    assert dataset in ('mnist')
    # to reproduce the same shuffle
    random.seed(0)
    # mnist = get_mnist()
    all_samples = get_all_samples(dataset, isize)
    logger.info('sample num: {}'.format(len(all_samples)))

    random.shuffle(all_samples)

    # folds = cfg.DATASET.FOLDS_COUNT
    folds = fold_num

    class_bins = {}

    for x in all_samples:
        if x[1] not in class_bins:
            class_bins[x[1]] = []
        class_bins[x[1]].append(x)

    sample_folds = [[] for _ in range(folds)]

    for _class, data in class_bins.items():
        count = len(data)
        logger.info("class %d count: %d" % (_class, count))

        count_per_fold = count // folds

        for i in range(folds):
            sample_folds[i] += data[i * count_per_fold: (i + 1) * count_per_fold]

    for i in range(len(sample_folds)):
        logger.info('fold {}, size {}'.format(i, len(sample_folds[i])))

    return sample_folds
