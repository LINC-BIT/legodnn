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