
import os
import re
from PIL import Image
# from .common.log import logger
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class Coil100AnomalyDataset(Dataset):
    def __init__(self, root_dir, anomaly_classes, train=True, transform=None):
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
            
            if train and label not in anomaly_classes and index not in test_indexes: # 训练模型 & 不属于异常类别 & 不是测试索引
                self.imgs += [img]
                self.labels += [0]
            if not train:
                if index not in test_indexes and label in anomaly_classes: # 不是测试索引且标签为异常类
                    self.imgs += [img]
                    self.labels += [1]
                    abnormal_num += 1
                if index in test_indexes:
                    self.imgs += [img]
                    self.labels += [label in anomaly_classes]
                    normal_num += int(label not in anomaly_classes)
                    abnormal_num += int(label in anomaly_classes)
        
        if train:
            print('train coil-100 dataset: {} normal samples'.format(len(self.labels)))
        else:
            print('test coil-100 dataset: {} normal samples, {} abnormal samples'.format(normal_num, abnormal_num))
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img, label = self.imgs[index], self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
        
def ganomaly_coil100_dataloader(dataroot='/data/datasets/', inlinears=None, outliners=list(range(50, 60)), train_batch_size=64, test_batch_size=64, image_size=32, num_workers=8):
    assert train_batch_size==test_batch_size
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    train_set = Coil100AnomalyDataset(os.path.join(dataroot, 'coil-100'), outliners, train=True, transform=transform)
    test_set = Coil100AnomalyDataset(os.path.join(dataroot, 'coil-100'), outliners, train=False, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    return train_loader, test_loader

# if __name__=='__main__':
    