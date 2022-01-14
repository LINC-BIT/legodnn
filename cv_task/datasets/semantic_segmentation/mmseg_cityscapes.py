
from mmseg.datasets import build_dataloader, build_dataset
                            
def mmseg_cityscapes_dataloader(cfg):
    lego_train_dataset = build_dataset(cfg.data.legotrain)
    test_dataset = build_dataset(cfg.data.test)

    lego_train_loader = build_dataloader(
        lego_train_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=True)

    test_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )    

    return lego_train_loader, test_loader

                            
def mmseg_cityscapes_dataloader_1213(cfg):
    train_dataset = build_dataset(cfg.data.train)
    test_dataset = build_dataset(cfg.data.test)

    lego_train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=True,
        seed=cfg.seed,
        drop_last=True)

    test_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )    

    return lego_train_loader, test_loader

    # data_loaders = [
    #     build_dataloader(
    #         ds,
    #         cfg.data.samples_per_gpu,
    #         cfg.data.workers_per_gpu,
    #         # cfg.gpus will be ignored if distributed
    #         len(cfg.gpu_ids),
    #         dist=distributed,
    #         seed=cfg.seed,
    #         drop_last=True) for ds in dataset
    # ]