# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from collections import OrderedDict

import numpy as np
from mmcv import Config

from ...builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class AnimalLocustDataset(Kpt2dSviewRgbImgTopDownDataset):
    """AnimalLocustDataset for animal pose estimation.

    `DeepPoseKit, a software toolkit for fast and robust animal
     pose estimation using deep learning'
    Elife'2019. More details can be found in the `paper.

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Desert Locust keypoint indexes::

        0: "head",
        1: "neck",
        2: "thorax",
        3: "abdomen1",
        4: "abdomen2",
        5: "anttipL",
        6: "antbaseL",
        7: "eyeL",
        8: "forelegL1",
        9: "forelegL2",
        10: "forelegL3",
        11: "forelegL4",
        12: "midlegL1",
        13: "midlegL2",
        14: "midlegL3",
        15: "midlegL4",
        16: "hindlegL1",
        17: "hindlegL2",
        18: "hindlegL3",
        19: "hindlegL4",
        20: "anttipR",
        21: "antbaseR",
        22: "eyeR",
        23: "forelegR1",
        24: "forelegR2",
        25: "forelegR3",
        26: "forelegR4",
        27: "midlegR1",
        28: "midlegR2",
        29: "midlegR3",
        30: "midlegR4",
        31: "hindlegR1",
        32: "hindlegR2",
        33: "hindlegR3",
        34: "hindlegR4"

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/locust.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.ann_info['use_different_joint_weights'] = False
        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        gt_db = []
        bbox_id = 0
        num_joints = self.ann_info['num_joints']
        for img_id in self.img_ids:

            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = self.coco.loadAnns(ann_ids)

            for obj in objs:
                if max(obj['keypoints']) == 0:
                    continue
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

                keypoints = np.array(obj['keypoints']).reshape(-1, 3)
                joints_3d[:, :2] = keypoints[:, :2]
                joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

                # the ori image is 160x160
                center, scale = self._xywh2cs(0, 0, 160, 160, 0.8)

                image_file = os.path.join(self.img_prefix,
                                          self.id2name[img_id])

                gt_db.append({
                    'image_file': image_file,
                    'center': center,
                    'scale': scale,
                    'rotation': 0,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'dataset': self.dataset_name,
                    'bbox': obj['bbox'],
                    'bbox_score': 1,
                    'bbox_id': bbox_id
                })
                bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db

    def evaluate(self, outputs, res_folder, metric='PCK', **kwargs):
        """Evaluate Fly keypoint results. The pose prediction results will be
        saved in `${res_folder}/result_keypoints.json`.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            outputs (list(preds, boxes, image_path, output_heatmap))
                :preds (np.ndarray[N,K,3]): The first two dimensions are
                    coordinates, score is the third dimension of the array.
                :boxes (np.ndarray[N,6]): [center[0], center[1], scale[0]
                    , scale[1],area, score]
                :image_paths (list[str]): For example, ['Test/source/0.jpg']
                :output_heatmap (np.ndarray[N, K, H, W]): model outputs.

            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed.
                Options: 'PCK', 'AUC', 'EPE'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'AUC', 'EPE']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')

        kpts = []
        for output in outputs:
            preds = output['preds']
            boxes = output['boxes']
            image_paths = output['image_paths']
            bbox_ids = output['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]

                kpts.append({
                    'keypoints': preds[i].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        return name_value
