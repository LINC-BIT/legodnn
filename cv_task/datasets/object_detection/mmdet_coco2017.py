from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
                            
def mmdet_coco2017_dataloader(cfg, train_batch_size=128, test_batch_size=128, num_workers=8):
    # cfg = Config.fromfile('/data/gxy/legodnn-public-version_object_detection/cv_task/object_detection/mmdet_models/configs/_base_/datasets/legodnn_coco_detection.py')
    # cfg = Config.fromfile(config)
    train_dataset = build_dataset(cfg.data.legotrain)
    # train_dataset = build_dataset(cfg.data.legotrain)
    test_dataset = build_dataset(cfg.data.test)

    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=train_batch_size,
        workers_per_gpu=num_workers,
        dist=False,
        shuffle=True)

    test_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=test_batch_size,
        workers_per_gpu=num_workers,
        dist=False,
        shuffle=False
    )    

    return train_loader, test_loader


