import os
import re
from PIL import Image
import numpy as np
# from .common.log import logger
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST

def ganomaly_emnist_dataloader(dataroot='/data/datasets/', inlinears=None, outliner='letters', train_batch_size=64, test_batch_size=64, image_size=32, num_workers=8):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    train_set = EMNIST(root=dataroot, split='bymerge', train=True, transform=transform, download=True)
    test_set = EMNIST(root=dataroot, split='bymerge', train=False, download=True, transform=transform)
    
    trn_img, trn_lbl = train_set.data, train_set.targets
    tst_img, tst_lbl = test_set.data, test_set.targets
        
    if outliner == 'letters':
        nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() < 10)[0])
        abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() >= 10)[0])
        nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() < 10)[0])
        abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() >= 10)[0])
    elif outliner == 'digits':
        nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() >= 10)[0])
        abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() < 10)[0])
        nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() >= 10)[0])
        abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() < 10)[0])
    else:
        abn_cls = int(outliner)
        nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() >= abn_cls)[0])
        abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() < abn_cls)[0])
        nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() >= abn_cls)[0])
        abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() < abn_cls)[0])
    
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    new_tst_img = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)
    
    print('train normal samples {}, test normal samples {}, test abnormal samples '
                '{}'.format(new_trn_img.size()[0], nrm_tst_img.size()[0], abn_trn_img.size()[0] + abn_tst_img.size()[0]))

    train_set.data = new_trn_img
    train_set.targets = new_trn_lbl
    test_set.data = new_tst_img
    test_set.targets = new_tst_lbl
    
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    return train_loader, test_loader