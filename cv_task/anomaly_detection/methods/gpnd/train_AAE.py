# Copyright 2018-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch.utils.data
from torch import optim
from torchvision.utils import save_image
from torch.autograd import Variable
import time
# import logging
import os
# from cv_task.anomaly_detection.methods.gpnd.dataloading import make_datasets, make_dataloader
from cv_task.anomaly_detection.methods.gpnd.net import GPND, Generator, Discriminator, Encoder, ZDiscriminator_mergebatch, ZDiscriminator
# from cv_task.anomaly_detection.methods.gpnd.test_AAE import test
# from cv_task.anomaly_detection.methods.gpnd.utils.tracker import LossTracker
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import copy
import random

import sys
sys.path.insert(0, '/data/zql/zedl')
from zedl.common.log import logger
from zedl.common.data_record import CSVDataRecord
from zedl.common.file import ensure_dir

sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/fn3')
from fn3_channel_open_api.fn3_channel import set_fn3_channel_channels, export_active_sub_net


def train(G, E, model_save_path, train_loader, nc, isize, zsize, epoch_num, batch_size=64, lr=0.002):

    # G = Generator(zsize, channels=nc)
    # G.weight_init(mean=0, std=0.02)
    G = G.cuda()

    D = Discriminator(channels=nc).cuda()
    D.weight_init(mean=0, std=0.02)

    # E = Encoder(zsize, channels=nc)
    # E.weight_init(mean=0, std=0.02)
    E = E.cuda()

    # if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH:
    #     ZD = ZDiscriminator_mergebatch(zsize, cfg.TRAIN.BATCH_SIZE)
    # else:
    #     ZD = ZDiscriminator(zsize, cfg.TRAIN.BATCH_SIZE)
    ZD = ZDiscriminator(zsize, batch_size).cuda()
    ZD.weight_init(mean=0, std=0.02)

    # lr = cfg.TRAIN.BASE_LEARNING_RATE

    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    GE_optimizer = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lr, betas=(0.5, 0.999))
    ZD_optimizer = optim.Adam(ZD.parameters(), lr=lr, betas=(0.5, 0.999))

    BCE_loss = nn.BCELoss()
    sample = torch.randn(64, zsize).view(-1, zsize, 1, 1)
    
    # ensure_dir(metrics_trending_csv_path)
    # ensure_dir(best_auc_g_save_path)
    # csv_data_record = CSVDataRecord(metrics_trending_csv_path, 
    #                                 ['epoch', 'auc', 'auprc', 'f1'])
    
    best_auc, best_auprc, best_f1 = 0., 0., 0.

    # tracker = LossTracker(output_folder=output_folder)

    for epoch in range(epoch_num):
        logger.info('epoch {} start'.format(epoch))

        G.train()
        D.train()
        E.train()
        ZD.train()

        epoch_start_time = time.time()

        # train_dataloader = make_dataloader(train_set, batch_size, torch.cuda.current_device())
        # train_set.shuffle()

        if (epoch + 1) % 30 == 0:
            G_optimizer.param_groups[0]['lr'] /= 4
            D_optimizer.param_groups[0]['lr'] /= 4
            GE_optimizer.param_groups[0]['lr'] /= 4
            ZD_optimizer.param_groups[0]['lr'] /= 4
            logger.info("epoch {}, learning rate change!".format(epoch))

        pbar = tqdm.tqdm(train_loader, dynamic_ncols=True)
        for x, y in pbar:
            x = x.to('cuda')
            y = y.to('cuda')
            
            x = x.view(-1, nc, isize, isize)

            y_real_ = torch.ones(x.shape[0]).cuda()
            y_fake_ = torch.zeros(x.shape[0]).cuda()

            # y_real_z = torch.ones(1 if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH else x.shape[0])
            # y_fake_z = torch.zeros(1 if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH else x.shape[0])
            y_real_z = torch.ones(x.shape[0]).cuda()
            y_fake_z = torch.zeros(x.shape[0]).cuda()

            #############################################

            D.zero_grad()

            D_result = D(x).squeeze()
            # print(D_result.size())
            D_real_loss = BCE_loss(D_result, y_real_)

            z = torch.randn((x.shape[0], zsize)).view(-1, zsize, 1, 1).cuda()
            z = Variable(z)

            x_fake = G(z).detach()
            D_result = D(x_fake).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)

            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()

            D_optimizer.step()

            # tracker.update(dict(D=D_train_loss))


            #############################################

            G.zero_grad()

            z = torch.randn((x.shape[0], zsize)).view(-1, zsize, 1, 1).cuda()
            z = Variable(z)

            x_fake = G(z)
            D_result = D(x_fake).squeeze()

            G_train_loss = BCE_loss(D_result, y_real_)

            G_train_loss.backward()
            G_optimizer.step()

            # tracker.update(dict(G=G_train_loss))

            #############################################

            ZD.zero_grad()

            z = torch.randn((x.shape[0], zsize)).view(-1, zsize).cuda()
            z = z.requires_grad_(True)

            ZD_result = ZD(z).squeeze()
            ZD_real_loss = BCE_loss(ZD_result, y_real_z)

            z = E(x).squeeze().detach()

            ZD_result = ZD(z).squeeze()
            ZD_fake_loss = BCE_loss(ZD_result, y_fake_z)

            ZD_train_loss = ZD_real_loss + ZD_fake_loss
            ZD_train_loss.backward()

            ZD_optimizer.step()

            # tracker.update(dict(ZD=ZD_train_loss))

            # #############################################

            E.zero_grad()
            G.zero_grad()

            z = E(x)
            x_d = G(z)

            ZD_result = ZD(z.squeeze()).squeeze()

            E_train_loss = BCE_loss(ZD_result, y_real_z) * 1.0

            Recon_loss = F.binary_cross_entropy(x_d, x.detach()) * 2.0

            (Recon_loss + E_train_loss).backward()

            GE_optimizer.step()

            # tracker.update(dict(GE=Recon_loss, E=E_train_loss))
            pbar.set_description('GE: {:.6f}, E: {:.6f}'.format(Recon_loss, E_train_loss))
            

            # #############################################

        # comparison = torch.cat([x, x_d])
        # save_image(comparison.cpu(), os.path.join(output_folder, 'reconstruction_' + str(epoch) + '.png'), nrow=x.shape[0])

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
    
    gpnd = GPND(zsize, nc)
    gpnd.e = E
    gpnd.g = G

    torch.save(gpnd, model_save_path)


def train_us_net(G, E, model_save_path, train_loader, nc, isize, zsize, epoch_num, batch_size=64, lr=0.002, 
                 train_width_mult_list=[0.125, 1], train_sample_net_num=4, cal=False):
    
    # G = Generator(zsize, channels=nc)
    # G.weight_init(mean=0, std=0.02)
    G = G.cuda()

    D = Discriminator(channels=nc).cuda()
    D.weight_init(mean=0, std=0.02)

    # E = Encoder(zsize, channels=nc)
    # E.weight_init(mean=0, std=0.02)
    E = E.cuda()

    # if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH:
    #     ZD = ZDiscriminator_mergebatch(zsize, cfg.TRAIN.BATCH_SIZE)
    # else:
    #     ZD = ZDiscriminator(zsize, cfg.TRAIN.BATCH_SIZE)
    ZD = ZDiscriminator(zsize, batch_size).cuda()
    ZD.weight_init(mean=0, std=0.02)

    # lr = cfg.TRAIN.BASE_LEARNING_RATE

    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    GE_optimizer = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lr, betas=(0.5, 0.999))
    ZD_optimizer = optim.Adam(ZD.parameters(), lr=lr, betas=(0.5, 0.999))

    BCE_loss = nn.BCELoss()
    sample = torch.randn(64, zsize).view(-1, zsize, 1, 1)
    
    # ensure_dir(metrics_trending_csv_path)
    # ensure_dir(best_auc_g_save_path)
    # csv_data_record = CSVDataRecord(metrics_trending_csv_path, 
    #                                 ['epoch', 'auc', 'auprc', 'f1'])
    
    best_auc, best_auprc, best_f1 = 0., 0., 0.

    # tracker = LossTracker(output_folder=output_folder)

    for epoch in range(epoch_num):
        logger.info('epoch {} start'.format(epoch))

        G.train()
        D.train()
        E.train()
        ZD.train()

        epoch_start_time = time.time()

        # train_dataloader = make_dataloader(train_set, batch_size, torch.cuda.current_device())
        # train_set.shuffle()

        if (epoch + 1) % 30 == 0:
            G_optimizer.param_groups[0]['lr'] /= 4
            D_optimizer.param_groups[0]['lr'] /= 4
            GE_optimizer.param_groups[0]['lr'] /= 4
            ZD_optimizer.param_groups[0]['lr'] /= 4
            logger.info("epoch {}, learning rate change!".format(epoch))

        pbar = tqdm.tqdm(train_loader, dynamic_ncols=True)
        for x, y in pbar:
            x = x.to('cuda')
            y = y.to('cuda')
            
            x = x.view(-1, nc, isize, isize)

            y_real_ = torch.ones(x.shape[0]).cuda()
            y_fake_ = torch.zeros(x.shape[0]).cuda()

            # y_real_z = torch.ones(1 if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH else x.shape[0])
            # y_fake_z = torch.zeros(1 if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH else x.shape[0])
            y_real_z = torch.ones(x.shape[0]).cuda()
            y_fake_z = torch.zeros(x.shape[0]).cuda()
            
            D.zero_grad()
            G.zero_grad()
            ZD.zero_grad()
            E.zero_grad()
            G.zero_grad()
            
            def cal_forward():
                z = E(x)
                x_d = G(z)

            #############################################
            def optimize():
                

                D_result = D(x).squeeze()
                # print(D_result.size())
                D_real_loss = BCE_loss(D_result, y_real_)

                z = torch.randn((x.shape[0], zsize)).view(-1, zsize, 1, 1).cuda()
                z = Variable(z)

                x_fake = G(z).detach()
                D_result = D(x_fake).squeeze()
                D_fake_loss = BCE_loss(D_result, y_fake_)

                D_train_loss = D_real_loss + D_fake_loss
                D_train_loss.backward()

                D_optimizer.step()

                # tracker.update(dict(D=D_train_loss))


                #############################################

                

                z = torch.randn((x.shape[0], zsize)).view(-1, zsize, 1, 1).cuda()
                z = Variable(z)

                x_fake = G(z)
                D_result = D(x_fake).squeeze()

                G_train_loss = BCE_loss(D_result, y_real_)

                G_train_loss.backward()
                G_optimizer.step()

                # tracker.update(dict(G=G_train_loss))

                #############################################

                

                z = torch.randn((x.shape[0], zsize)).view(-1, zsize).cuda()
                z = z.requires_grad_(True)

                ZD_result = ZD(z).squeeze()
                ZD_real_loss = BCE_loss(ZD_result, y_real_z)

                z = E(x).squeeze().detach()

                ZD_result = ZD(z).squeeze()
                ZD_fake_loss = BCE_loss(ZD_result, y_fake_z)

                ZD_train_loss = ZD_real_loss + ZD_fake_loss
                ZD_train_loss.backward()

                ZD_optimizer.step()

                # tracker.update(dict(ZD=ZD_train_loss))

                # #############################################

                

                z = E(x)
                x_d = G(z)

                ZD_result = ZD(z.squeeze()).squeeze()

                E_train_loss = BCE_loss(ZD_result, y_real_z) * 1.0

                Recon_loss = F.binary_cross_entropy(x_d, x.detach()) * 2.0

                (Recon_loss + E_train_loss).backward()

                GE_optimizer.step()

                # tracker.update(dict(GE=Recon_loss, E=E_train_loss))
                pbar.set_description('GE: {:.6f}, E: {:.6f}'.format(Recon_loss, E_train_loss))
            

            if cal:
                for width_mult in widths_train:
                    G.apply(lambda m: setattr(m, 'width_mult', width_mult))
                    E.apply(lambda m: setattr(m, 'width_mult', width_mult))
                
                    cal_forward()
                continue
            
            min_width, max_width = min(train_width_mult_list), max(train_width_mult_list)
            widths_train = []
            for _ in range(train_sample_net_num - 2):
                widths_train.append(
                    random.uniform(min_width, max_width))
                
            widths_train = [max_width, min_width] + widths_train
            # logger.info('sampled width: {}'.format(widths_train))

            for width_mult in widths_train:
                G.apply(lambda m: setattr(m, 'width_mult', width_mult))
                E.apply(lambda m: setattr(m, 'width_mult', width_mult))
                
                optimize()
            
            # #############################################

        # comparison = torch.cat([x, x_d])
        # save_image(comparison.cpu(), os.path.join(output_folder, 'reconstruction_' + str(epoch) + '.png'), nrow=x.shape[0])

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
    
    if not cal:
        gpnd = GPND(zsize, nc)
        gpnd.e = E
        gpnd.g = G

        torch.save(gpnd, model_save_path)


def train_fn3_channel(G, E, model_save_path, train_loader, nc, isize, zsize, epoch_num, batch_size=64, lr=0.002,
                      fn3_channel_g_key_layers_info=[], fn3_channel_e_key_layers_info=[],
                      fn3_channel_g_disabled_layers=[], fn3_channel_e_disabled_layers=[]):
    
    # G = Generator(zsize, channels=nc)
    # G.weight_init(mean=0, std=0.02)
    G = G.cuda()

    D = Discriminator(channels=nc).cuda()
    D.weight_init(mean=0, std=0.02)

    # E = Encoder(zsize, channels=nc)
    # E.weight_init(mean=0, std=0.02)
    E = E.cuda()

    # if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH:
    #     ZD = ZDiscriminator_mergebatch(zsize, cfg.TRAIN.BATCH_SIZE)
    # else:
    #     ZD = ZDiscriminator(zsize, cfg.TRAIN.BATCH_SIZE)
    ZD = ZDiscriminator(zsize, batch_size).cuda()
    ZD.weight_init(mean=0, std=0.02)

    # lr = cfg.TRAIN.BASE_LEARNING_RATE

    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    GE_optimizer = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lr, betas=(0.5, 0.999))
    ZD_optimizer = optim.Adam(ZD.parameters(), lr=lr, betas=(0.5, 0.999))

    BCE_loss = nn.BCELoss()
    sample = torch.randn(64, zsize).view(-1, zsize, 1, 1)
    
    # ensure_dir(metrics_trending_csv_path)
    # ensure_dir(best_auc_g_save_path)
    # csv_data_record = CSVDataRecord(metrics_trending_csv_path, 
    #                                 ['epoch', 'auc', 'auprc', 'f1'])
    
    best_auc, best_auprc, best_f1 = 0., 0., 0.

    # tracker = LossTracker(output_folder=output_folder)

    for epoch in range(epoch_num):
        logger.info('epoch {} start'.format(epoch))

        G.train()
        D.train()
        E.train()
        ZD.train()

        epoch_start_time = time.time()

        # train_dataloader = make_dataloader(train_set, batch_size, torch.cuda.current_device())
        # train_set.shuffle()

        if (epoch + 1) % 30 == 0:
            G_optimizer.param_groups[0]['lr'] /= 4
            D_optimizer.param_groups[0]['lr'] /= 4
            GE_optimizer.param_groups[0]['lr'] /= 4
            ZD_optimizer.param_groups[0]['lr'] /= 4
            logger.info("epoch {}, learning rate change!".format(epoch))

        pbar = tqdm.tqdm(train_loader, dynamic_ncols=True)
        for x, y in pbar:
            x = x.to('cuda')
            y = y.to('cuda')
            
            x = x.view(-1, nc, isize, isize)

            y_real_ = torch.ones(x.shape[0]).cuda()
            y_fake_ = torch.zeros(x.shape[0]).cuda()

            # y_real_z = torch.ones(1 if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH else x.shape[0])
            # y_fake_z = torch.zeros(1 if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH else x.shape[0])
            y_real_z = torch.ones(x.shape[0]).cuda()
            y_fake_z = torch.zeros(x.shape[0]).cuda()

            #############################################
            def optimize():
                D.zero_grad()

                D_result = D(x).squeeze()
                # print(D_result.size())
                D_real_loss = BCE_loss(D_result, y_real_)

                z = torch.randn((x.shape[0], zsize)).view(-1, zsize, 1, 1).cuda()
                z = Variable(z)

                x_fake = G(z).detach()
                D_result = D(x_fake).squeeze()
                D_fake_loss = BCE_loss(D_result, y_fake_)

                D_train_loss = D_real_loss + D_fake_loss
                D_train_loss.backward()

                D_optimizer.step()

                # tracker.update(dict(D=D_train_loss))


                #############################################

                G.zero_grad()

                z = torch.randn((x.shape[0], zsize)).view(-1, zsize, 1, 1).cuda()
                z = Variable(z)

                x_fake = G(z)
                D_result = D(x_fake).squeeze()

                G_train_loss = BCE_loss(D_result, y_real_)

                G_train_loss.backward()
                G_optimizer.step()

                # tracker.update(dict(G=G_train_loss))

                #############################################

                ZD.zero_grad()

                z = torch.randn((x.shape[0], zsize)).view(-1, zsize).cuda()
                z = z.requires_grad_(True)

                ZD_result = ZD(z).squeeze()
                ZD_real_loss = BCE_loss(ZD_result, y_real_z)

                z = E(x).squeeze().detach()

                ZD_result = ZD(z).squeeze()
                ZD_fake_loss = BCE_loss(ZD_result, y_fake_z)

                ZD_train_loss = ZD_real_loss + ZD_fake_loss
                ZD_train_loss.backward()

                ZD_optimizer.step()

                # tracker.update(dict(ZD=ZD_train_loss))

                # #############################################

                E.zero_grad()
                G.zero_grad()

                z = E(x)
                x_d = G(z)

                ZD_result = ZD(z.squeeze()).squeeze()

                E_train_loss = BCE_loss(ZD_result, y_real_z) * 1.0

                Recon_loss = F.binary_cross_entropy(x_d, x.detach()) * 2.0

                (Recon_loss + E_train_loss).backward()

                GE_optimizer.step()

                # tracker.update(dict(GE=Recon_loss, E=E_train_loss))
                pbar.set_description('GE: {:.6f}, E: {:.6f}'.format(Recon_loss, E_train_loss))
            
            
            fn3_channel_g_layers_name = [i[0] for i in fn3_channel_g_key_layers_info]
            fn3_channel_g_channels = [i[1] for i in fn3_channel_g_key_layers_info]
            fn3_channel_e_layers_name = [i[0] for i in fn3_channel_e_key_layers_info]
            fn3_channel_e_channels = [i[1] for i in fn3_channel_e_key_layers_info]
            
            for i, c in enumerate(fn3_channel_g_channels):
                if fn3_channel_g_layers_name[i] in fn3_channel_g_disabled_layers:
                    continue
                fn3_channel_g_channels[i] = random.randint(1, c)
            set_fn3_channel_channels(G, fn3_channel_g_channels)
            
            for i, c in enumerate(fn3_channel_e_channels):
                if fn3_channel_e_layers_name[i] in fn3_channel_e_disabled_layers:
                    continue
                fn3_channel_e_channels[i] = random.randint(1, c)
            set_fn3_channel_channels(E, fn3_channel_e_channels)
            
            optimize()
            
            # #############################################

        # comparison = torch.cat([x, x_d])
        # save_image(comparison.cpu(), os.path.join(output_folder, 'reconstruction_' + str(epoch) + '.png'), nrow=x.shape[0])

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
    
    gpnd = GPND(zsize, nc)
    gpnd.e = E
    gpnd.g = G

    torch.save(gpnd, model_save_path)


def train_ofa_channel(G, E, model_save_path, train_loader, nc, isize, zsize, epoch_num, batch_size=64, lr=0.002, train_width_mult_list=[1.0]):
    
    # G = Generator(zsize, channels=nc)
    # G.weight_init(mean=0, std=0.02)
    G = G.cuda()

    D = Discriminator(channels=nc).cuda()
    D.weight_init(mean=0, std=0.02)

    # E = Encoder(zsize, channels=nc)
    # E.weight_init(mean=0, std=0.02)
    E = E.cuda()

    # if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH:
    #     ZD = ZDiscriminator_mergebatch(zsize, cfg.TRAIN.BATCH_SIZE)
    # else:
    #     ZD = ZDiscriminator(zsize, cfg.TRAIN.BATCH_SIZE)
    ZD = ZDiscriminator(zsize, batch_size).cuda()
    ZD.weight_init(mean=0, std=0.02)

    # lr = cfg.TRAIN.BASE_LEARNING_RATE

    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    GE_optimizer = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lr, betas=(0.5, 0.999))
    ZD_optimizer = optim.Adam(ZD.parameters(), lr=lr, betas=(0.5, 0.999))

    BCE_loss = nn.BCELoss()
    sample = torch.randn(64, zsize).view(-1, zsize, 1, 1)
    
    # ensure_dir(metrics_trending_csv_path)
    # ensure_dir(best_auc_g_save_path)
    # csv_data_record = CSVDataRecord(metrics_trending_csv_path, 
    #                                 ['epoch', 'auc', 'auprc', 'f1'])
    
    best_auc, best_auprc, best_f1 = 0., 0., 0.

    # tracker = LossTracker(output_folder=output_folder)

    for epoch in range(epoch_num):
        logger.info('epoch {} start'.format(epoch))

        G.train()
        D.train()
        E.train()
        ZD.train()

        epoch_start_time = time.time()

        # train_dataloader = make_dataloader(train_set, batch_size, torch.cuda.current_device())
        # train_set.shuffle()

        if (epoch + 1) % 30 == 0:
            G_optimizer.param_groups[0]['lr'] /= 4
            D_optimizer.param_groups[0]['lr'] /= 4
            GE_optimizer.param_groups[0]['lr'] /= 4
            ZD_optimizer.param_groups[0]['lr'] /= 4
            logger.info("epoch {}, learning rate change!".format(epoch))

        pbar = tqdm.tqdm(train_loader, dynamic_ncols=True)
        for x, y in pbar:
            x = x.to('cuda')
            y = y.to('cuda')
            
            x = x.view(-1, nc, isize, isize)

            y_real_ = torch.ones(x.shape[0]).cuda()
            y_fake_ = torch.zeros(x.shape[0]).cuda()

            # y_real_z = torch.ones(1 if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH else x.shape[0])
            # y_fake_z = torch.zeros(1 if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH else x.shape[0])
            y_real_z = torch.ones(x.shape[0]).cuda()
            y_fake_z = torch.zeros(x.shape[0]).cuda()

            #############################################
            def optimize():
                D.zero_grad()

                D_result = D(x).squeeze()
                # print(D_result.size())
                D_real_loss = BCE_loss(D_result, y_real_)

                z = torch.randn((x.shape[0], zsize)).view(-1, zsize, 1, 1).cuda()
                z = Variable(z)

                x_fake = G(z).detach()
                D_result = D(x_fake).squeeze()
                D_fake_loss = BCE_loss(D_result, y_fake_)

                D_train_loss = D_real_loss + D_fake_loss
                D_train_loss.backward()

                D_optimizer.step()

                # tracker.update(dict(D=D_train_loss))


                #############################################

                G.zero_grad()

                z = torch.randn((x.shape[0], zsize)).view(-1, zsize, 1, 1).cuda()
                z = Variable(z)

                x_fake = G(z)
                D_result = D(x_fake).squeeze()

                G_train_loss = BCE_loss(D_result, y_real_)

                G_train_loss.backward()
                G_optimizer.step()

                # tracker.update(dict(G=G_train_loss))

                #############################################

                ZD.zero_grad()

                z = torch.randn((x.shape[0], zsize)).view(-1, zsize).cuda()
                z = z.requires_grad_(True)

                ZD_result = ZD(z).squeeze()
                ZD_real_loss = BCE_loss(ZD_result, y_real_z)

                z = E(x).squeeze().detach()

                ZD_result = ZD(z).squeeze()
                ZD_fake_loss = BCE_loss(ZD_result, y_fake_z)

                ZD_train_loss = ZD_real_loss + ZD_fake_loss
                ZD_train_loss.backward()

                ZD_optimizer.step()

                # tracker.update(dict(ZD=ZD_train_loss))

                # #############################################

                E.zero_grad()
                G.zero_grad()

                z = E(x)
                x_d = G(z)

                ZD_result = ZD(z.squeeze()).squeeze()

                E_train_loss = BCE_loss(ZD_result, y_real_z) * 1.0

                Recon_loss = F.binary_cross_entropy(x_d, x.detach()) * 2.0

                (Recon_loss + E_train_loss).backward()

                GE_optimizer.step()

                # tracker.update(dict(GE=Recon_loss, E=E_train_loss))
                pbar.set_description('GE: {:.6f}, E: {:.6f}'.format(Recon_loss, E_train_loss))
            
            
            sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/once-for-all')
            from ofa_open_api.ofa_net import sample_gpnd_ofa_sub_net_width_mult
            sample_gpnd_ofa_sub_net_width_mult(G, train_width_mult_list)
            sample_gpnd_ofa_sub_net_width_mult(E, train_width_mult_list)
            optimize()
            
            # #############################################

        # comparison = torch.cat([x, x_d])
        # save_image(comparison.cpu(), os.path.join(output_folder, 'reconstruction_' + str(epoch) + '.png'), nrow=x.shape[0])

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
    
    gpnd = GPND(zsize, nc)
    gpnd.e = E
    gpnd.g = G

    torch.save(gpnd, model_save_path)


sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/cgnet')
from cgnet_open_api import convert_model_to_cgnet, add_cgnet_loss
def train_cgnet(G, E, model_save_path, train_loader, nc, isize, zsize, epoch_num, batch_size=64, lr=0.002, gtar=0):
    
    # G = Generator(zsize, channels=nc)
    # G.weight_init(mean=0, std=0.02)
    G = G.cuda()

    D = Discriminator(channels=nc).cuda()
    D.weight_init(mean=0, std=0.02)

    # E = Encoder(zsize, channels=nc)
    # E.weight_init(mean=0, std=0.02)
    E = E.cuda()

    # if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH:
    #     ZD = ZDiscriminator_mergebatch(zsize, cfg.TRAIN.BATCH_SIZE)
    # else:
    #     ZD = ZDiscriminator(zsize, cfg.TRAIN.BATCH_SIZE)
    ZD = ZDiscriminator(zsize, batch_size).cuda()
    ZD.weight_init(mean=0, std=0.02)

    # lr = cfg.TRAIN.BASE_LEARNING_RATE

    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    GE_optimizer = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lr, betas=(0.5, 0.999))
    ZD_optimizer = optim.Adam(ZD.parameters(), lr=lr, betas=(0.5, 0.999))

    BCE_loss = nn.BCELoss()
    sample = torch.randn(64, zsize).view(-1, zsize, 1, 1)
    
    # ensure_dir(metrics_trending_csv_path)
    # ensure_dir(best_auc_g_save_path)
    # csv_data_record = CSVDataRecord(metrics_trending_csv_path, 
    #                                 ['epoch', 'auc', 'auprc', 'f1'])
    
    best_auc, best_auprc, best_f1 = 0., 0., 0.

    # tracker = LossTracker(output_folder=output_folder)

    for epoch in range(epoch_num):
        logger.info('epoch {} start'.format(epoch))

        G.train()
        D.train()
        E.train()
        ZD.train()

        epoch_start_time = time.time()

        # train_dataloader = make_dataloader(train_set, batch_size, torch.cuda.current_device())
        # train_set.shuffle()

        if (epoch + 1) % 30 == 0:
            G_optimizer.param_groups[0]['lr'] /= 4
            D_optimizer.param_groups[0]['lr'] /= 4
            GE_optimizer.param_groups[0]['lr'] /= 4
            ZD_optimizer.param_groups[0]['lr'] /= 4
            logger.info("epoch {}, learning rate change!".format(epoch))

        pbar = tqdm.tqdm(train_loader, dynamic_ncols=True)
        for x, y in pbar:
            x = x.to('cuda')
            y = y.to('cuda')
            
            x = x.view(-1, nc, isize, isize)

            y_real_ = torch.ones(x.shape[0]).cuda()
            y_fake_ = torch.zeros(x.shape[0]).cuda()

            # y_real_z = torch.ones(1 if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH else x.shape[0])
            # y_fake_z = torch.zeros(1 if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH else x.shape[0])
            y_real_z = torch.ones(x.shape[0]).cuda()
            y_fake_z = torch.zeros(x.shape[0]).cuda()

            #############################################

            D.zero_grad()

            D_result = D(x).squeeze()
            # print(D_result.size())
            D_real_loss = BCE_loss(D_result, y_real_)

            z = torch.randn((x.shape[0], zsize)).view(-1, zsize, 1, 1).cuda()
            z = Variable(z)

            x_fake = G(z).detach()
            D_result = D(x_fake).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)

            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()

            D_optimizer.step()

            # tracker.update(dict(D=D_train_loss))


            #############################################

            G.zero_grad()

            z = torch.randn((x.shape[0], zsize)).view(-1, zsize, 1, 1).cuda()
            z = Variable(z)

            x_fake = G(z)
            D_result = D(x_fake).squeeze()

            G_train_loss = BCE_loss(D_result, y_real_)
            add_cgnet_loss(G, G_train_loss, gtar)

            G_train_loss.backward()
            G_optimizer.step()

            # tracker.update(dict(G=G_train_loss))

            #############################################

            ZD.zero_grad()

            z = torch.randn((x.shape[0], zsize)).view(-1, zsize).cuda()
            z = z.requires_grad_(True)

            ZD_result = ZD(z).squeeze()
            ZD_real_loss = BCE_loss(ZD_result, y_real_z)

            z = E(x).squeeze().detach()

            ZD_result = ZD(z).squeeze()
            ZD_fake_loss = BCE_loss(ZD_result, y_fake_z)

            ZD_train_loss = ZD_real_loss + ZD_fake_loss
            ZD_train_loss.backward()

            ZD_optimizer.step()

            # tracker.update(dict(ZD=ZD_train_loss))

            # #############################################

            E.zero_grad()
            G.zero_grad()

            z = E(x)
            x_d = G(z)

            ZD_result = ZD(z.squeeze()).squeeze()

            E_train_loss = BCE_loss(ZD_result, y_real_z) * 1.0
            add_cgnet_loss(E, E_train_loss, gtar)

            Recon_loss = F.binary_cross_entropy(x_d, x.detach()) * 2.0

            (Recon_loss + E_train_loss).backward()

            GE_optimizer.step()

            # tracker.update(dict(GE=Recon_loss, E=E_train_loss))
            pbar.set_description('GE: {:.6f}, E: {:.6f}'.format(Recon_loss, E_train_loss))
            

            # #############################################

        # comparison = torch.cat([x, x_d])
        # save_image(comparison.cpu(), os.path.join(output_folder, 'reconstruction_' + str(epoch) + '.png'), nrow=x.shape[0])

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
    
    gpnd = GPND(zsize, nc)
    gpnd.e = E
    gpnd.g = G

    torch.save(gpnd, model_save_path)

