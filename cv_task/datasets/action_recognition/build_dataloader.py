from mmaction.datasets import build_dataloader, build_dataset
                            
def mmaction_build_dataloader(cfg):
    train_dataset = build_dataset(cfg.data.train)
    # test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    test_dataset = build_dataset(cfg.data.test)

    train_loader = build_dataloader(
        train_dataset,
        videos_per_gpu=cfg.data.videos_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        seed=cfg.seed,
        shuffle=True,
        drop_last=True)

    test_loader = build_dataloader(
        test_dataset,
        videos_per_gpu=cfg.data.videos_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        drop_last=False
    )    

    return train_loader, test_loader
