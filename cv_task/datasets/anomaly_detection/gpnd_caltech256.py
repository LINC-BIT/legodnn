import os
import re
from PIL import Image
import numpy as np
# from .common.log import logger
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import Caltech256

class Caltech256AnomalyDataset(Caltech256):
        def __init__(self, root, inliners, outliners, transform=None, target_transform=None, download=False):
            super(Caltech256AnomalyDataset, self).__init__(root, transform, target_transform, download)
            
            def is_normal(i):
                return i in inliners
            def is_abnormal(i):
                return i in outliners
            
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
            
            self._train_normal_imgs_path = []
            self._test_normal_imgs_path = []
            self._abnormal_imgs_path = []
            
            self._class_num = np.zeros((257, ))
            for item in self.y:
                self._class_num[item] += 1
            # print(np.argsort(self._class_num)[::-1])
            
            self._normal_samples_num, self._abnormal_samples_num = 0, 0
            
            for index in range(len(self.y)):
                img_path = os.path.join(self.root,
                                        "256_ObjectCategories",
                                        self.categories[self.y[index]],
                                        "{:03d}_{:04d}.jpg".format(self.y[index] + 1, self.index[index]))
                label = self.y[index]
                
                if is_normal(label):
                    self._imgs_path += [img_path]
                    if self.index[index] < self._class_num[self.y[index]] * 0.8:
                        self._train_normal_imgs_path += [img_path]
                    else:
                        self._test_normal_imgs_path += [img_path]
                    self._labels += [0]
                    self._normal_samples_num += 1
                if is_abnormal(label):
                    self._imgs_path += [img_path]
                    self._abnormal_imgs_path += [img_path]
                    self._labels += [1]
                    self._abnormal_samples_num += 1
             
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
        
def gpnd_caltech256_dataloader(dataroot='/data/datasets/', inliners=[250, 144, 252, 22, 67], outliners=[256], train_batch_size=64, test_batch_size=64, image_size=32, num_workers=8):

    transform = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.Lambda(lambda img: img / 255.)
    ])
    dataset = Caltech256AnomalyDataset(os.path.join(dataroot, './Caltech-256/data/'), inliners, outliners, transform=transform)
    
    train_normal_imgs_path, test_normal_imgs_path = dataset._train_normal_imgs_path, dataset._test_normal_imgs_path
    abnormal_imgs_path = dataset._abnormal_imgs_path
    
    if len(test_normal_imgs_path) > len(abnormal_imgs_path):
        test_normal_imgs_path = test_normal_imgs_path[:len(abnormal_imgs_path)]
    else:
        abnormal_imgs_path = abnormal_imgs_path[:len(test_normal_imgs_path)]
    
    train_set = Caltech256AnomalyDataset(os.path.join(dataroot, './Caltech-256/data/'), inliners, outliners, transform=transform)
    train_set._imgs_path = train_normal_imgs_path
    train_set._labels = [0 for _ in train_normal_imgs_path]
    
    test_set = Caltech256AnomalyDataset(os.path.join(dataroot, './Caltech-256/data/'), inliners, outliners, transform=transform)
    test_set._imgs_path = test_normal_imgs_path + abnormal_imgs_path
    test_set._labels = [0 for _ in test_normal_imgs_path] + [1 for _ in abnormal_imgs_path]
    
    print('train normal sample {}, test normal sample {}, '
                'test abnormal sample {}'.format(len(train_normal_imgs_path), len(test_normal_imgs_path), len(abnormal_imgs_path)))
    
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    return train_loader, test_loader