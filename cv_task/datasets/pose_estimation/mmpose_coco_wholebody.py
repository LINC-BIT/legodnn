from mmpose.datasets import build_dataloader, build_dataset
                            
def mmpose_coco_wholebody_dataloader(cfg, train_batch_size=128, test_batch_size=128, num_workers=1):
    train_dataset = build_dataset(cfg.data.legotrain)
    # test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
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


