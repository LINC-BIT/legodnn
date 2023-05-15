from pickle import NONE
import time
import os
import copy
import random
import torch
from .utils import progress_bar, format_time
from cv_task.utils.data_record_v2 import DataRecord
import sys
import torch.nn.functional as F
sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/fn3')
from fn3_channel_open_api_1215.fn3_channel import set_fn3_channel_channels, export_active_sub_net

sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/cgnet')
from cgnet_open_api_1212 import convert_model_to_cgnet, add_cgnet_loss, get_cgnet_flops_save_ratio, get_cgnet_flops_save_ratio

from legodnn.utils.dl.common.model import get_module
from baselines.nested_network.nestdnn_1230.nestdnn_open_api import zero_grads_nestdnn_layers

# Training
def train_one_epoch(net, criterion, optimizer, epoch, train_loader, device='cuda', **kwargs):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start = time.time()
    method = kwargs.get('method', None)
    original_kwargs = copy.deepcopy(kwargs)
        
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if method is not None:
            if method=='usnet':
                width_mult_list = kwargs.get('width_mult_list', None)
                sample_net_num = kwargs.get('sample_net_num', None)
                assert width_mult_list is not None
                assert sample_net_num is not None
                min_width, max_width = min(width_mult_list), max(width_mult_list)
                widths_train = []
                for _ in range(sample_net_num - 2):
                    widths_train.append(
                        random.uniform(min_width, max_width))
                widths_train = [max_width, min_width] + widths_train
                
                optimizer.zero_grad()
                for width in widths_train:
                    # print(width)
                    net.apply(lambda m: setattr(m, 'width_mult', width))
                    outputs = net(inputs)
                    loss = criterion(outputs, targets) # compute loss
                    loss.backward()  # backward
                optimizer.step()
                net.apply(lambda m: setattr(m, 'width_mult', 1.0))
            
            elif method=='fn3':
                fn3_all_layers = original_kwargs.get('fn3_all_layers', None)
                fn3_disable_layers = original_kwargs.get('fn3_disable_layers', None)
                min_sparsity = original_kwargs.get('min_sparsity', None)
                
                assert fn3_all_layers is not None
                assert fn3_disable_layers is not None
                
                fn3_channel_layers_name = [i[0] for i in fn3_all_layers]
                fn3_channel_channels = [i[1] for i in fn3_all_layers]
                # print(fn3_channel_channels)
                # print(type(fn3_channel_channels[0]))
                for i, c in enumerate(fn3_channel_channels):
                    if fn3_channel_layers_name[i] in fn3_disable_layers:
                        continue
                    fn3_channel_channels[i] = random.randint(int(min_sparsity*c), c)
                    
                set_fn3_channel_channels(net, fn3_channel_channels)
                
                optimizer.zero_grad()  # zero grad
                outputs = net(inputs)# forward fn3
                loss = criterion(outputs, targets)# compute loss, step
                loss.backward() # backward
                optimizer.step()
                
                set_fn3_channel_channels(net, [i[1] for i in fn3_all_layers])
                
            elif method=='nestdnn':
                grad_positions = original_kwargs.get('grad_positions', None)
                assert grad_positions is not None
                
                optimizer.zero_grad()  # zero grad
                outputs = net(inputs)# forward fn3
                loss = criterion(outputs, targets)# compute loss, step
                loss.backward() # backward
                
                for key, values in grad_positions.items():
                    nest_module = get_module(net, key)
                    for value_key, value in values.items():
                        if value_key.startswith('weight'):
                            assert len(value)==1 or len(value)==4
                            if len(value)==1:
                                nest_module.weight.grad[:value[0]].data.zero_()
                            elif len(value)==4:
                                nest_module.weight.grad[:value[0], :value[1], :value[2], :value[3]].data.zero_()
                            else:
                                raise NotImplementedError
                            
                        elif value_key.startswith('bias'):
                            assert len(value)==1
                            nest_module.bias.grad[:value[0]].data.zero_()
                        else:
                            raise NotImplementedError
                        # exit(0)
                    pass
                
                optimizer.step()  # step
            elif method=='nestdnnv3':
                zero_shape_info = original_kwargs.get('zero_shape_info', None)
                assert zero_shape_info is not None
                
                optimizer.zero_grad()  # zero grad
                outputs = net(inputs)# forward fn3
                loss = criterion(outputs, targets)# compute loss, step
                loss.backward() # backward

                zero_grads_nestdnn_layers(net, zero_shape_info) # 清空前一个模型的梯度
                optimizer.step()  # step
                
            elif method=='ofa':
                teacher_model = original_kwargs.get('teacher_model', None)
                depth_list = original_kwargs.get('depth_list', None)
                width_list = original_kwargs.get('width_list', None)
                expand_list = original_kwargs.get('expand_list', None)
                kd_loss = original_kwargs.get('kd_loss', None)
                kd_ratio = original_kwargs.get('kd_ratio', None)
                assert teacher_model is not None
                assert depth_list is not None
                assert width_list is not None
                assert expand_list is not None
                assert kd_loss is not None
                assert kd_ratio is not None
                optimizer.zero_grad()
                net.sample_active_subnet()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                if kd_ratio>0:
                    with torch.no_grad():
                        soft_logits = teacher_model(inputs).detach()
                        soft_label = F.softmax(soft_logits, dim=1)
                    kd_loss = kd_loss(outputs, soft_label)
                    
                    loss += kd_ratio*kd_loss
                loss.backward()
                optimizer.step()
                net.set_max_net()
            else:
                raise NotImplementedError
        else:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Train: Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    train_time = format_time(time.time()-start)
    train_report = 'Epoch %d: Train time: %s, Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, train_time, train_loss/(len(train_loader)), 100.*correct/total, correct, total)
    return train_report, train_loss/(len(train_loader)), 100.*correct/total

def _test_model(net, criterion, epoch, test_loader, data_record, device, **kwargs):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Test: Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    test_time = format_time(time.time()-start)
    test_report = 'Test time: %s, Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_time, test_loss/(len(test_loader)), 100.*correct/total, correct, total)

    # Save checkpoint.
    acc = 100.*correct/total
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    torch.save(state, data_record.model_save_file + '.pth')
    return test_report, test_loss/(len(test_loader)), 100.*correct/total

def train_model(net, config_dict, train_loader, test_loader, model_save_file, device='cuda',  **kwargs):
    # net = net.to(device)
    data_record = DataRecord(model_save_file)
    data_record.record_opt(config_dict)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=config_dict['learning_rate'],
                        momentum=config_dict['momentum'], weight_decay=config_dict['weight_decay'])
    epoch_num=config_dict['epoch_num']
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config_dict['milestones'], gamma=config_dict['gamma'])
    
    test_report, test_loss, test_acc = _test_model(net, criterion, -1, test_loader, data_record, device, **kwargs)
    data_record.record_report(report_str=(test_report))
    for epoch in range(epoch_num):
        train_report, train_loss, train_acc = train_one_epoch(net, criterion, optimizer, epoch, train_loader, device, **kwargs)
        test_report, test_loss, test_acc = _test_model(net, criterion, epoch, test_loader, data_record, device, **kwargs)
        data_record.record_report(report_str=(train_report + '  ' +test_report))
        data_record.state_dict_update([('train_loss_list', train_loss),
                                    ('train_acc_list', train_acc),
                                    ('test_loss_list', test_loss),
                                    ('test_acc_list', test_acc)])
        scheduler.step()