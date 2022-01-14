"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import MNIST, FashionMNIST, EMNIST, KMNIST
from torchvision.datasets import CIFAR10, Caltech256, SVHN, CIFAR100
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import re

from utils.log import logger
from cv_task.anomaly_detection.methods.ganomaly.options import Options

# more reusable
def load_data_easily(dataset, abnormal_class, isize, batch_size):
    opt = Options().parse()
    opt.dataset = dataset
    opt.abnormal_class = abnormal_class
    opt.isize = isize
    opt.batchsize = batch_size
    
    return load_data(opt)

##
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)

    if opt.dataset.startswith('cifar10'):
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': False}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        classes = {
            'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }

        dataset = {}
        dataset['train'] = CIFAR10(root=opt.dataroot, train=True, download=True, transform=transform)
        dataset['test'] = CIFAR10(root=opt.dataroot, train=False, download=True, transform=transform)

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_cifar_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            abn_cls_idx=classes[opt.abnormal_class],
            manualseed=opt.manualseed
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader
    
    if opt.dataset.startswith('caltech256'):
        anomaly_classes_range = [int(i) for i in opt.abnormal_class.split('_')]
        anomaly_classes = list(range(anomaly_classes_range[0], anomaly_classes_range[1]))
        
        train_loader, test_loader = Caltech256AnomalyDetectionDataLoader(os.path.join(opt.dataroot, './Caltech-256/data/'), anomaly_classes, opt.isize, opt.batchsize)
        return dict(train=train_loader, test=test_loader)

    elif opt.dataset in ['mnist']:
        opt.abnormal_class = int(opt.abnormal_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        dataset = {}
        dataset['train'] = MNIST(root=opt.dataroot, train=True, download=True, transform=transform)
        dataset['test'] = MNIST(root=opt.dataroot, train=False, download=True, transform=transform)

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_mnist_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            abn_cls_idx=opt.abnormal_class,
            manualseed=opt.manualseed
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader
    
    elif opt.dataset in ['fmnist']:
        opt.abnormal_class = int(opt.abnormal_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        dataset = {}
        dataset['train'] = FashionMNIST(root=opt.dataroot, train=True, download=True, transform=transform)
        dataset['test'] = FashionMNIST(root=opt.dataroot, train=False, download=True, transform=transform)

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_mnist_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            abn_cls_idx=opt.abnormal_class,
            manualseed=opt.manualseed
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader
    
    elif opt.dataset in ['coil100']:
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': False}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        dataset = {}
        anomaly_classes_range = [int(i) for i in opt.abnormal_class.split('_')]
        anomaly_classes = list(range(anomaly_classes_range[0], anomaly_classes_range[1]))
        dataset['train'] = Coil100AnomalyDataset(opt.dataroot, anomaly_classes, train=True, transform=transform)
        dataset['test'] = Coil100AnomalyDataset(opt.dataroot, anomaly_classes, train=False, transform=transform)

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader
    
    elif opt.dataset.startswith('imagenet2012'):
        anomaly_classes_range = [int(i) for i in opt.abnormal_class.split('_')]
        anomaly_classes = list(range(anomaly_classes_range[0], anomaly_classes_range[1]))
        
        train_loader, test_loader = ImageNetAnomalyDetectionDataLoader(os.path.join(opt.dataroot, './imagenet2012/ILSVRC2012_img_train'), 
                                                                       os.path.join(opt.dataroot, './imagenet2012/ILSVRC2012_img_val'), 
                                                                       anomaly_classes, opt.isize, opt.batchsize)
        return {
            'train': train_loader,
            'test': test_loader
        }

    elif opt.dataset in ['svhn']:
        opt.abnormal_class = int(opt.abnormal_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        dataset = {}
        dataset['train'] = SVHN(root=opt.dataroot, split='train', download=True, transform=transform)
        dataset['test'] = SVHN(root=opt.dataroot, split='test', download=True, transform=transform)

        dataset['train'].data, dataset['train'].labels, \
        dataset['test'].data, dataset['test'].labels = get_svhn_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].labels,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].labels,
            abn_cls_idx=opt.abnormal_class,
            manualseed=opt.manualseed
        ) # the same with MNIST

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader
    
    elif opt.dataset.startswith('emnist'):
        # opt.abnormal_class = int(opt.abnormal_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        dataset = {}
        dataset['train'] = EMNIST(root=opt.dataroot, split='bymerge', train=True, transform=transform, download=True)
        dataset['test'] = EMNIST(root=opt.dataroot, split='bymerge', train=False, download=True, transform=transform)
        # print(set(dataset['test'].targets))
        # print(dataset['train'].classes)
        # print(dataset['train'].class_to_idx)
        
        # assert opt.abnormal_class in ['digits', 'letters']

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_emnist_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            abn_cls=opt.abnormal_class,
            manualseed=opt.manualseed
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader
    
    elif opt.dataset in ['kmnist']:
        # opt.abnormal_class = int(opt.abnormal_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        dataset = {}
        dataset['train'] = KMNIST(root=opt.dataroot, train=True, download=True, transform=transform)
        dataset['test'] = KMNIST(root=opt.dataroot, train=False, download=True, transform=transform)

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_mnist_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            abn_cls_idx=opt.abnormal_class,
            manualseed=opt.manualseed
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader
    
    elif opt.dataset.startswith('cifar100-'):
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': False}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        # classes = {
        #     'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
        #     'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        # }
        anomaly_classes_range = [int(i) for i in opt.abnormal_class.split('_')]
        anomaly_classes = list(range(anomaly_classes_range[0], anomaly_classes_range[1]))

        dataset = {}
        dataset['train'] = CIFAR100(root=opt.dataroot, train=True, download=True, transform=transform)
        dataset['test'] = CIFAR100(root=opt.dataroot, train=False, download=True, transform=transform)

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_cifar100_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            abn_cls_idx=anomaly_classes,
            manualseed=opt.manualseed
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader
    
    else:
        raise NotImplementedError()

##
def get_cifar_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # Convert train-test labels into numpy array.
    trn_lbl = np.array(trn_lbl)
    tst_lbl = np.array(tst_lbl)

    # --
    # Find idx, img, lbl for abnormal and normal on org dataset.
    nrm_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

    nrm_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # --
    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = np.concatenate((nrm_trn_img, nrm_tst_img), axis=0)
        nrm_lbl = np.concatenate((nrm_trn_lbl, nrm_tst_lbl), axis=0)
        abn_img = np.concatenate((abn_trn_img, abn_tst_img), axis=0)
        abn_lbl = np.concatenate((abn_trn_lbl, abn_tst_lbl), axis=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal
    new_trn_img = np.copy(nrm_trn_img)
    new_trn_lbl = np.copy(nrm_trn_lbl)
    new_tst_img = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

##
def get_mnist_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # --
    # Find normal abnormal indexes.
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != abn_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == abn_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != abn_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == abn_cls_idx)[0])

    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # --
    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = torch.cat((nrm_trn_img, nrm_tst_img), dim=0)
        nrm_lbl = torch.cat((nrm_trn_lbl, nrm_tst_lbl), dim=0)
        abn_img = torch.cat((abn_trn_img, abn_tst_img), dim=0)
        abn_lbl = torch.cat((abn_trn_lbl, abn_tst_lbl), dim=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    # Create new anomaly dataset based on the following data structure:
    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    new_tst_img = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

##
def get_mnist2_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, nrm_cls_idx=0, proportion=0.5,
                               manualseed=-1):
    """ Create mnist 2 anomaly dataset.

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        nrm_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [tensor] -- New training-test images and labels.
    """
    # Seed for deterministic behavior
    if manualseed != -1:
        torch.manual_seed(manualseed)

    # --
    # Find normal abnormal indexes.
    # TODO: PyTorch v0.4 has torch.where function
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == nrm_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != nrm_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == nrm_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != nrm_cls_idx)[0])

    # Get n percent of the abnormal samples.
    abn_tst_idx = abn_tst_idx[torch.randperm(len(abn_tst_idx))]
    abn_tst_idx = abn_tst_idx[:int(len(abn_tst_idx) * proportion)]


    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    new_tst_img = torch.cat((nrm_tst_img, abn_tst_img), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_tst_lbl), dim=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl


class Coil100AnomalyDataset(Dataset):
    def __init__(self, root_dir, anomaly_classes, train=True, transform=None):
        root_dir = os.path.join(root_dir, 'coil-100')
        self.transform = transform
        
        test_indexes = [300, 315, 15, 305, 180, 135, 170, 245, 275, 335, 330, 270]
        
        regx = re.compile('obj(\\d+)__(\\d+)\.png')
        
        self.imgs, self.labels = [], []
        
        normal_num, abnormal_num = 0, 0
        for img_file_path in os.listdir(root_dir):
            if not img_file_path.endswith('.png'):
                continue
            
            img_file_path = os.path.join(root_dir, img_file_path)
            
            img = Image.open(img_file_path).convert('RGB')
            
            label, index = regx.findall(img_file_path)[0]
            label, index = int(label), int(index)
            
            if train and label not in anomaly_classes and index not in test_indexes:
                self.imgs += [img]
                self.labels += [0]
            if not train:
                if index not in test_indexes and label in anomaly_classes:
                    self.imgs += [img]
                    self.labels += [1]
                    abnormal_num += 1
                if index in test_indexes:
                    self.imgs += [img]
                    self.labels += [label in anomaly_classes]
                    normal_num += int(label not in anomaly_classes)
                    abnormal_num += int(label in anomaly_classes)
        
        if train:
            logger.info('train coil-100 dataset: {} normal samples'.format(len(self.labels)))
        else:
            logger.info('test coil-100 dataset: {} normal samples, {} abnormal samples'.format(normal_num, abnormal_num))
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img, label = self.imgs[index], self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def ImageNetAnomalyDetectionDataLoader(train_data_root, test_data_root, abnormal_class_indexes, img_size, batch_size):
    num_workers = 8
    
    normal_class_indexes = [430, 432, 429]
    
    # train loader
    train_transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        # transforms.RandomResizedCrop(img_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = ImageFolder(train_data_root, transform=train_transform)
    
    test_transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        # transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = ImageFolder(train_data_root, transform=train_transform)

    # modify train_dataset
    train_samples = []
    test_samples = []
    for sample in train_dataset.samples:
        p, class_idx = sample
        if class_idx in normal_class_indexes: # TODO
            train_samples += [(p, 0)]
        elif class_idx in abnormal_class_indexes:
            test_samples += [(p, 1)]
            
    # import math
    # ratio = math.round(len(train_samples) / len(test_samples) - 1)
    
    train_dataset.classes = ['normal']
    train_dataset.class_to_idx = [0]
    train_dataset.samples = train_samples[::2]
    train_dataset.targets = [s[1] for s in train_dataset.samples]
    
    logger.info('ImageNet anomaly train dataset: {} normal samples'.format(len(train_dataset.samples)))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    test_dataset.classes = ['normal', 'abnormal']
    test_dataset.class_to_idx = [0, 1]
    test_dataset.samples = train_samples[1::2] + test_samples
    test_dataset.targets = [s[1] for s in test_dataset.samples]
    
    logger.info('ImageNet anomaly test dataset: {} normal samples, {} abnormal samples'.format(len(train_samples[1::2]), len(test_samples)))
    
    # raise NotImplementedError()
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    return train_loader, test_loader


def Caltech256AnomalyDetectionDataLoader(root, abnormal_class_indexes, img_size, batch_size):
    class Caltech256AnomalyDataset(Caltech256):
        def __init__(self, root, abnormal_class_indexes, transform=None, target_transform=None, download=False, train=True):
            super(Caltech256AnomalyDataset, self).__init__(root, transform, target_transform, download)
            
            def is_normal(i):
                # return i not in abnormal_class_index
                return i in [250, 144, 252, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 72, 70, 200, 201, 202]
            
            self.index = []
            self.y = []
            for (i, c) in enumerate(self.categories):
                n = len(list(
                    filter(
                        lambda p: p.endswith('jpg'), 
                        os.listdir(os.path.join(self.root, "256_ObjectCategories", c))
                    )
                ))
                self.index.extend(range(1, n + 1))
                self.y.extend(n * [i])
            
            self._imgs_path = []
            self._labels = []
            
            self._class_num = np.zeros((257, ))
            for item in self.y:
                self._class_num[item] += 1
                
            # print(np.argsort(self._class_num)[::-1])
            
            # self._test_indexes = [24, 26, 72, 38, 52, 14, 47, 21, 62, 4, 80, 68, 32, 49, 46, 71, 6, 25, 27, 29, 28, 66, 30, 58, 5, 54, 20, 3, 15, 44, 76, 36, 73, 37, 13, 31, 1, 69, 53, 43, 70, 51, 7, 77, 67, 57, 34, 65, 35, 19, 59, 60, 79, 55, 8, 61, 2, 33, 9, 41, 78, 45, 12, 63, 16]
            # print(self.y[0])
            # print(self.categories)
            # print(self.index)
            
            self._normal_samples_num, self._abnormal_samples_num = 0, 0
            
            if not train:
                normal_imgs_path = []
                abnormal_imgs_path = []
            
            for index in range(len(self.y)):
                img_path = os.path.join(self.root,
                                        "256_ObjectCategories",
                                        self.categories[self.y[index]],
                                        "{:03d}_{:04d}.jpg".format(self.y[index] + 1, self.index[index]))
                label = self.y[index]
                
                if train:
                    if is_normal(label) and self.index[index] < self._class_num[self.y[index]] * 0.5:
                        self._imgs_path += [img_path]
                        self._labels += [0]
                        self._normal_samples_num += 1
                        
                else:
                    if is_normal(label) and self.index[index] >= self._class_num[self.y[index]] * 0.5:
                        self._imgs_path += [img_path]
                        self._labels += [0]
                        
                        # normal_imgs_path += [img_path]
                        
                        self._normal_samples_num += 1
                    if label in abnormal_class_indexes:
                        self._imgs_path += [img_path]
                        self._labels += [1]
                        # abnormal_imgs_path += [img_path]
                        
                        self._abnormal_samples_num += 1
            
            # if not train:
            #     import random
            #     random.shuffle(normal_imgs_path)
            #     normal_imgs_path = normal_imgs_path[:len(abnormal_imgs_path)]
            #     self._imgs_path = normal_imgs_path + abnormal_imgs_path
            #     self._labels = [0 for _ in normal_imgs_path] + [1 for _ in abnormal_imgs_path]
            
            logger.info('abnormal classes: {}'.format(abnormal_class_indexes))
            if train:
                logger.info('train set: normal samples {}'.format(self._normal_samples_num))
            else:
                logger.info('test set: normal samples {}, abnormal samples {}'.format(len(normal_imgs_path), len(abnormal_imgs_path)))
             
        def __getitem__(self, index):
            img = Image.open(self._imgs_path[index])
            
            img_np = np.array(img)
            if len(img_np.shape) < 3:
                t = np.empty((3, ) + img_np.shape, dtype=img_np.dtype)
                t[0] = np.array(img_np)
                t[1] = np.array(img_np)
                t[2] = np.array(img_np)
                img = Image.fromarray(t.transpose(1, 2, 0))
            
            target = self._labels[index]

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target
        
        def __len__(self):
            return len(self._imgs_path)
            
        
    num_workers = 8
    
    # train loader
    train_transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = Caltech256AnomalyDataset(root, abnormal_class_indexes, transform=train_transform, download=True, train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    # test loader
    test_transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = Caltech256AnomalyDataset(root, abnormal_class_indexes, transform=train_transform, download=True, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    logger.info('Caltech256 anomaly train samples: {}, test samples: {}'.format(len(train_dataset), len(test_dataset)))
    
    return train_loader, test_loader


def get_svhn_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # --
    # Find normal abnormal indexes.
    trn_lbl = np.array(trn_lbl)
    tst_lbl = np.array(tst_lbl)

    # --
    # Find idx, img, lbl for abnormal and normal on org dataset.
    nrm_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

    nrm_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # --
    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = np.concatenate((nrm_trn_img, nrm_tst_img), axis=0)
        nrm_lbl = np.concatenate((nrm_trn_lbl, nrm_tst_lbl), axis=0)
        abn_img = np.concatenate((abn_trn_img, abn_tst_img), axis=0)
        abn_lbl = np.concatenate((abn_trn_lbl, abn_tst_lbl), axis=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal
    new_trn_img = np.copy(nrm_trn_img)
    new_trn_lbl = np.copy(nrm_trn_lbl)
    new_tst_img = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)
    
    logger.info('SVHN - train samples {}, test samples {}'.format(new_trn_img.shape[0], new_tst_img.shape[0]))

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl


def get_emnist_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # --
    # Find normal abnormal indexes.
    
    if abn_cls == 'letters':
        nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() < 10)[0])
        abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() >= 10)[0])
        nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() < 10)[0])
        abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() >= 10)[0])
    elif abn_cls == 'digits':
        nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() >= 10)[0])
        abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() < 10)[0])
        nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() >= 10)[0])
        abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() < 10)[0])
    else:
        abn_cls = int(abn_cls)
        nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() >= abn_cls)[0])
        abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() < abn_cls)[0])
        nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() >= abn_cls)[0])
        abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() < abn_cls)[0])
    
    # nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != abn_cls_idx)[0])
    # abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == abn_cls_idx)[0])
    # nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != abn_cls_idx)[0])
    # abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == abn_cls_idx)[0])

    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # --
    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = torch.cat((nrm_trn_img, nrm_tst_img), dim=0)
        nrm_lbl = torch.cat((nrm_trn_lbl, nrm_tst_lbl), dim=0)
        abn_img = torch.cat((abn_trn_img, abn_tst_img), dim=0)
        abn_lbl = torch.cat((abn_trn_lbl, abn_tst_lbl), dim=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    # Create new anomaly dataset based on the following data structure:
    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    new_tst_img = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)
    
    logger.info('EMNIST - train normal samples {}, test normal / abnormal samples '
                '{} / {}'.format(new_trn_img.size()[0], nrm_tst_img.size()[0], abn_trn_img.size()[0] + abn_tst_img.size()[0]))

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl


def get_cifar100_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # Convert train-test labels into numpy array.
    trn_lbl = np.array(trn_lbl)
    tst_lbl = np.array(tst_lbl)
    
    def find_index(labels):
        normal_indexes = []
        abnormal_indexes = []
        
        for i, label in enumerate(labels):
            if label in abn_cls_idx:
                abnormal_indexes += [i]
            else:
                normal_indexes += [i]
                
        return normal_indexes, abnormal_indexes        

    # --
    # Find idx, img, lbl for abnormal and normal on org dataset.
    # nrm_trn_idx = np.where(trn_lbl not in abn_cls_idx)[0]
    # abn_trn_idx = np.where(trn_lbl in abn_cls_idx)[0]
    nrm_trn_idx, abn_trn_idx = find_index(trn_lbl)
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

    # nrm_tst_idx = np.where(tst_lbl not in abn_cls_idx)[0]
    # abn_tst_idx = np.where(tst_lbl in abn_cls_idx)[0]
    nrm_tst_idx, abn_tst_idx = find_index(tst_lbl)
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # --
    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = np.concatenate((nrm_trn_img, nrm_tst_img), axis=0)
        nrm_lbl = np.concatenate((nrm_trn_lbl, nrm_tst_lbl), axis=0)
        abn_img = np.concatenate((abn_trn_img, abn_tst_img), axis=0)
        abn_lbl = np.concatenate((abn_trn_lbl, abn_tst_lbl), axis=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal
    new_trn_img = np.copy(nrm_trn_img)
    new_trn_lbl = np.copy(nrm_trn_lbl)
    new_tst_img = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)
    
    logger.info('CIFAR100 - train normal samples {}, test normal / abnormal samples '
                '{} / {}'.format(new_trn_img.shape[0], nrm_tst_img.shape[0], abn_trn_img.shape[0] + abn_tst_img.shape[0]))

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl
