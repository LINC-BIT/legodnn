import torch
import tqdm
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

def CIFAR10Dataloader(root_dir='/data/zql/datasets/', train_batch_size=128, test_batch_size=128, num_workers=4):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    train_set = CIFAR10(root=root_dir, train=True, download=True, transform=transform_train)
    test_set = CIFAR10(root=root_dir, train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def CIFAR100Dataloader(root_dir='/data/zql/datasets/', train_batch_size=128, test_batch_size=128, num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=root_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR100(
        root=root_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def transforms_block_train(
        img_size=32,
        scale=(0.08, 1.0),
        color_jitter=0.4,
        interpolation='random',
        random_erasing=0.4,
        random_erasing_mode='const',
        use_prefetcher=False,
        mean=None,
        std=None
):
    from .transforms import RandomResizedCropAndInterpolation
    from .random_erasing import RandomErasing
    
    if isinstance(color_jitter, (list, tuple)):
        # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
        # or 4 if also augmenting hue
        assert len(color_jitter) in (3, 4)
    else:
        # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
        color_jitter = (float(color_jitter),) * 3

    tfl = [
        RandomResizedCropAndInterpolation(
            img_size, scale=scale, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(*color_jitter),
    ]

    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        # tfl += [ToNumpy()]
        raise NotImplementedError
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
        if random_erasing > 0.:
            tfl.append(RandomErasing(random_erasing, mode=random_erasing_mode, device='cpu'))
    return transforms.Compose(tfl)

def CIFAR100AugDataloader(root_dir='/data/zql/datasets/', train_batch_size=128, test_batch_size=128, num_workers=4):
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    # ])

    color_jitter = 0.4
    transform_train = transforms_block_train(32, color_jitter=color_jitter, interpolation='random', mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=root_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR100(
        root=root_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader