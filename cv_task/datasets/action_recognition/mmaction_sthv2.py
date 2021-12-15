from mmaction.datasets import build_dataloader, build_dataset

def mmaction_sthv2_dataloader(cfg, train_batch_size=128, test_batch_size=128, num_workers=1):
    cfg.data.legotrain.test_mode = True
    cfg.data.test.test_mode = True
    train_dataset = build_dataset(cfg.data.legotrain, dict(test_mode=True))
    test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))

    train_dataloader_setting = dict(
        videos_per_gpu=train_batch_size,
        workers_per_gpu=num_workers,
        dist=False,
        shuffle=False)
    train_dataloader_setting = dict(train_dataloader_setting,
                              **dict(videos_per_gpu=train_batch_size))
    train_loader = build_dataloader(
        train_dataset, 
        **train_dataloader_setting)
    
    test_dataloader_setting = dict(
        videos_per_gpu=test_batch_size,
        workers_per_gpu=num_workers,
        dist=False,
        shuffle=False)  
    test_dataloader_setting = dict(test_dataloader_setting,
                              **dict(videos_per_gpu=test_batch_size))
    test_loader = build_dataloader(
        test_dataset,
        **test_dataloader_setting
    )    

    return train_loader, test_loader