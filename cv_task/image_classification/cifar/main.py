'''Train CIFAR10 with PyTorch.'''
from math import gamma
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
import sys
sys.path.insert(0, '../../../')

from cv_task.image_classification.cifar.models import *
from cv_task.image_classification.cifar.models import mobilenetv2_w1, mobilenetv2_w2, vgg16, ran_92_32
from cv_task.image_classification.cifar.utils import progress_bar, format_time
from cv_task.utils.data_record import DataRecord
from cv_task.datasets.image_classification.data_loader import CIFAR10Dataloader, CIFAR100Dataloader
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default='ran_92_32', type=str, help='train model name [resnet18, inceptionv3, senet18, cbam_resnet18, resnext29_2x64d, wideresnet_40_10, mobilenetv2_w1, mobilenetv2_w2, vgg16, ran_92_32]')
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset [cifar10, cifar100]')
parser.add_argument('--root_path', default='/data/gxy/legodnn-public-version_9.27/cv_task_model/image_classification', type=str)

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=400, type=int, help='train epoch numbers')
parser.add_argument('--train_batchsize', default=128, type=int, help='train batchSize')
parser.add_argument('--test_batchsize', default=128, type=int, help='test batchSize')
parser.add_argument('--num_workers', default=8, type=int, help='train and test dataloader num_workers')

parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')

args = parser.parse_args()

if 'mobilenetv2' in args.model:
    args.wd = 4e-5

# args.model = 'mobilenetv2_w0p45'
# args.wd = 4e-5

# args.model = 'RAN'
# args.wd = 4e-5

data_record = DataRecord(args.root_path, args.dataset, args.model)
data_record.record_opt(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


if args.dataset=='cifar10':
    # Data
    print('==> Preparing data..')
    trainloader, testloader = CIFAR10Dataloader(batch_size=args.train_batchsize, num_workers=args.num_workers)
elif args.dataset=='cifar100':
    # Data
    print('==> Preparing data..')
    trainloader, testloader = CIFAR100Dataloader(batch_size=args.train_batchsize, num_workers=args.num_workers)
else:
    raise NotImplemented

# Model
print('==> Building model..')
print('train model: ', args.model)

if args.dataset=='cifar10':
    net = eval(args.model)(10)
elif args.dataset=='cifar100':
    net = eval(args.model)(100)
else:
    raise NotImplemented

net = net.to(device)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=8e-5)
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=args.wd)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print(targets.shape)
        # exit(0)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Train: Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    train_time = format_time(time.time()-start)
    train_report = 'Epoch %d: Train time: %s, Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, train_time, train_loss/(len(trainloader)), 100.*correct/total, correct, total)
    return train_report, train_loss/(len(trainloader)), 100.*correct/total
    data_record.record_report(report_str=train_report)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Test: Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    test_time = format_time(time.time()-start)
    test_report = 'Test time: %s, Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_time, test_loss/(len(testloader)), 100.*correct/total, correct, total)
    # data_record.record_report(report_str=test_report)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        torch.save(state, data_record._log_path + '/' + args.model + '.pth')
        best_acc = acc
        test_report += '  ----save model'

    return test_report, test_loss/(len(testloader)), 100.*correct/total
    
for epoch in range(start_epoch, args.epoch):
    train_report, train_loss, train_acc = train(epoch)
    test_report, test_loss, test_acc = test(epoch)
    data_record.record_report(report_str=(train_report + '  ' +test_report))
    data_record.state_dict_update([('train_loss_list', train_loss),
                                   ('train_acc_list', train_acc),
                                   ('test_loss_list', test_loss),
                                   ('test_acc_list', test_acc)])
    scheduler.step()
