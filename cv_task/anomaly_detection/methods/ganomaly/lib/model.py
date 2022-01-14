"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
from operator import sub
import os
import time
import numpy as np
from tqdm import tqdm
import random
from torch.autograd import Variable
import torch.optim as optim
import copy
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from cv_task.anomaly_detection.methods.ganomaly.lib.networks import NetG, NetD, weights_init
from cv_task.anomaly_detection.methods.ganomaly.lib.visualizer import Visualizer
from cv_task.anomaly_detection.methods.ganomaly.lib.loss import l2_loss
from cv_task.anomaly_detection.methods.ganomaly.lib.evaluate import evaluate, evaluate_all

import sys
sys.path.insert(0, '/data/zql/zedl')
from zedl.common.log import logger
from zedl.dl.common.model import save_model, ModelSaveMethod

sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/fn3')
from fn3_channel_open_api.fn3_channel import set_fn3_channel_channels, export_active_sub_net

sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/cgnet')
from cgnet_open_api import convert_model_to_cgnet, add_cgnet_loss, get_cgnet_flops_save_ratio


class BaseModel(torch.nn.Module):
    """ Base Model for ganomaly
    """
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        ##
        # Seed for deterministic behavior
        # self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        # self.visualizer = Visualizer(opt)
        # self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
        
        # self.metrics_record = []

    ##
    def set_input(self, input:torch.Tensor):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])

    ##
    # def seed(self, seed_value):
    #     """ Seed 
        
    #     Arguments:
    #         seed_value {int} -- [description]
    #     """
    #     # Check if seed is default value
    #     if seed_value == -1:
    #         return

    #     # Otherwise seed all functionality
    #     import random
    #     random.seed(seed_value)
    #     torch.manual_seed(seed_value)
    #     torch.cuda.manual_seed_all(seed_value)
    #     np.random.seed(seed_value)
    #     torch.backends.cudnn.deterministic = True
        # set_random_seed(seed_value)

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_enc', self.err_g_enc.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch, p):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        # weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        # if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        # torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
        #            '%s/netG.pth' % (weight_dir))
        # torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
        #            '%s/netD.pth' % (weight_dir))
        
        # torch.save(self.netg, p)
        if hasattr(self.opt, 'ofa'):
            save_model(self.netg, p+'.netg', ModelSaveMethod.FULL)
            save_model(self.netd, p+'.netd', ModelSaveMethod.FULL)
        else:
            save_model(self.netg, p, ModelSaveMethod.FULL)
            # save_model(export_active_sub_net(self.netg), p + '.jit', ModelSaveMethod.JIT, (1, self.opt.nc, self.opt.isize, self.opt.isize))

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        if (hasattr(self.opt, 'cal') and not self.opt.cal) or not hasattr(self.opt, 'cal'):
            # print("train one epoch")
            self.netg.train()
            # by queyu
            self.netd.train()
        
        epoch_iter = 0
        pbar = tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train']), dynamic_ncols=True)
        for data in pbar:
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            # self.optimize()
            self.optimize_params()
            
            if hasattr(self.opt, 'cal') and self.opt.cal:
                continue
            
            errors = dict(self.get_errors())
            pbar.set_description('d: {:.6f}, g: {:.6f}, g_adv: {:.6f}, g_con: {:.6f}, '
                                 'g_enc: {:.6f}'.format(errors['err_d'], errors['err_g'], errors['err_g_adv'], 
                                                        errors['err_g_con'], errors['err_g_enc']))

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    # self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                pass
                # reals, fakes, fixed = self.get_current_images()
                # self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                # if self.opt.display:
                    # self.visualizer.display_current_images(reals, fakes, fixed)
                    

        # print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))
        # self.visualizer.print_current_errors(self.epoch, errors)

    ##
    def train_model(self, dataloader):
        """ Train the model
        """
        self.dataloader = dataloader
        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0
        best_auprc = 0
        best_f1 = 0
        
        # self.csv_data_record = CSVDataRecord(self.opt.metrics_trending_csv_path, 
        #                                      ['epoch', 'auc', 'auprc', 'f1'])

        # Train for niter epochs.
        # print(">> Training model %s." % self.name)
        logger.info('start training...')
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()
            
            # if hasattr(self.opt, 'ofa') and self.opt.ofa:
            #     (auc, auprc, f1) = self.test_ofa_model(dataloader)
            # else:
            #     (auc, auprc, f1), _ = self.test_model(dataloader)
            
            (auc, auprc, f1), _ = self.test_model(dataloader)
            # self.metrics_record += [(auc, auprc, f1)]
            # self.csv_data_record.write([self.epoch, auc, auprc, f1])
            # print(111111111111111111111111111111)
            self.save_weights(self.epoch, self.opt.model_save_path)
            
            from zedl.common.data_record import write_json
            if getattr(self.opt, 'cg_net', 'False'):
                write_json( self.opt.model_save_path + '.metrics',
                           {'auc': auc, 'sparsity': self.cgnet_reduced_flops_ratio }, backup=False)
                
            if auc > best_auc:
                best_auc = auc
                self.save_weights(self.epoch, self.opt.model_save_path + '.best')
                if getattr(self.opt, 'cg_net', 'False'):
                    write_json(self.opt.model_save_path + '.best.metrics',
                               {'auc': best_auc, 'sparsity': self.cgnet_reduced_flops_ratio }, 
                                backup=False)
            # print(222222222222222222222222222222222222)
                # logger.info('save best auc model in {}'.format(self.opt.best_auc_g_save_path))
            # if auprc > best_auprc:
            #     best_auprc = auprc
            #     self.save_weights(self.epoch, self.opt.best_auprc_g_save_path)
            #     logger.info('save best auprc model in {}'.format(self.opt.best_auprc_g_save_path))
            # if f1 > best_f1:
            #     best_f1 = f1
            #     self.save_weights(self.epoch, self.opt.best_f1_g_save_path)
            #     logger.info('save best f1 model in {}'.format(self.opt.best_f1_g_save_path))
            
            # self.visualizer.print_current_performance(res, best_auc)
        # print(">> Training model %s.[Done]" % self.name)

    ##
    def test_ofa_model(self, dataloader):
        self.netg.eval()
        self.netd.eval()
        sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/once-for-all')
        from ofa_open_api.ofa_net import set_ganomaly_ofa_sub_net_width_mult
        from ofa_open_api.utils import set_running_statistics
        sub_net_of_auc, sub_net_of_auprc, sub_net_of_f1  = [], [], []
        for width_mult in self.opt.train_width_mult_list:
            set_ganomaly_ofa_sub_net_width_mult(self.netg, width_mult)
            start_time__ = time.time()
            set_running_statistics(self.netg, dataloader['train'])
            (auc, auprc, f1), _ = self.test_model(dataloader)
            print("test ofa sub net W{0} auc: {1}, time: {2}s".format(width_mult, auc, (time.time()-start_time__)))
            sub_net_of_auc.append(auc)
            sub_net_of_auprc.append(auprc)
            sub_net_of_f1.append(f1)
        print('average auc: {}'.format(np.mean(np.array(sub_net_of_auc))))
        return np.mean(np.array(sub_net_of_auc)), np.mean(np.array(sub_net_of_auprc)), np.mean(np.array(sub_net_of_f1))
            
    def test_model(self, dataloader):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        self.dataloader = dataloader
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'
            
            # by queyu
            self.netg.eval()
            self.netd.eval()

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            # print("   Testing model %s." % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            pbar = tqdm(self.dataloader['test'], dynamic_ncols=True, leave=False)
            
            self.cgnet_reduced_flops_ratio = 0
            batch_num = 0
            
            for i, data in enumerate(pbar, 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input.to(self.device))
                
                if getattr(self.opt, 'cg_net', False):
                    cur_reduced_flops_ratio = get_cgnet_flops_save_ratio(self.netg, self.input)
                    self.cgnet_reduced_flops_ratio += cur_reduced_flops_ratio

                error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)
                time_o = time.time()

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                self.latent_o [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)

                self.times.append(time_o - time_i)
                batch_num += 1

                # Save test images.
                # if self.opt.save_test_images:
                #     dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                #     if not os.path.isdir(dst):
                #         os.makedirs(dst)
                #     real, fake, _ = self.get_current_images()
                #     vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                #     vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)

            if getattr(self.opt, 'cg_net', 'False'):
                self.cgnet_reduced_flops_ratio /= batch_num
                logger.info('cgnet - reduced {:.2f}% flops'.format(self.cgnet_reduced_flops_ratio * 100.0))
            
            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.raw_an_scores = copy.deepcopy(self.an_scores)
            
            # by queyu
            # for const test
            if torch.max(self.an_scores) == torch.min(self.an_scores):
                if torch.max(self.an_scores) == 1:
                    logger.info('make an_scores = 1')
                    self.an_scores = torch.ones_like(self.an_scores)
                elif torch.max(self.an_scores) == 0:
                    logger.info('make an_scores = 0')
                    self.an_scores = torch.zeros_like(self.an_scores)
                else:
                    logger.error('unexpected behavior')
                    raise NotImplementedError()
            else:
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
                
            # by queyu
            # get histogram data of normal/abnormal test samples
            normal_samples_index = torch.nonzero(torch.eq(self.gt_labels, 0)).squeeze(dim=1)
            abnormal_samples_index = torch.nonzero(torch.eq(self.gt_labels, 1)).squeeze(dim=1)
            
            normal_samples_score = self.an_scores[normal_samples_index].detach().cpu().numpy().tolist()
            abnormal_samples_score = self.an_scores[abnormal_samples_index].detach().cpu().numpy().tolist()
            
            raw_normal_samples_score = self.raw_an_scores[normal_samples_index].detach().cpu().numpy().tolist()
            raw_abnormal_samples_score = self.raw_an_scores[abnormal_samples_index].detach().cpu().numpy().tolist()
                
            # auc, eer = roc(self.gt_labels, self.an_scores)
            # auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            
            auc, auprc, f1 = evaluate_all(self.gt_labels, self.an_scores)
            logger.info('epoch [{}/{}] - auc: {:.6f}, auprc: {:.6f}, f1: {:.6f}'.format(self.epoch, self.opt.niter, auc, auprc, f1))
            
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), (self.opt.metric, auc)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                # self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return (auc, auprc, f1), (-1, -1)

##
class Ganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self): return 'Ganomaly'

    def __init__(self, opt):
        super(Ganomaly, self).__init__(opt)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones (size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
    
        
    def reset_optimizer(self):
        # self.netg.apply(weights_init)
        # self.netd.apply(weights_init)
        self.netg.to(self.device)
        self.netd.to(self.device)
        self.netg.train()
        self.netd.train()
        # print(self.opt.lr)
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake, self.latent_i, self.latent_o = self.netg(self.input.to(self.device))

    ##
    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input.to(self.device))
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())

    ##
    def backward_g(self):
        """ Backpropagate through netG
        """
        self.err_g_adv = self.l_adv(self.netd(self.input.to(self.device))[1], self.netd(self.fake.to(self.device))[1])
        self.err_g_con = self.l_con(self.fake.to(self.device), self.input.to(self.device))
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)
        self.err_g = self.err_g_adv * self.opt.w_adv + \
                     self.err_g_con * self.opt.w_con + \
                     self.err_g_enc * self.opt.w_enc
                     
                     
        if getattr(self.opt, 'cg_net', False):
            add_cgnet_loss(self.netg, self.err_g, self.opt.gtar)
                     
        self.err_g.backward(retain_graph=True)

    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss
        self.err_d_real = self.l_bce(self.pred_real.to(self.device), self.real_label.to(self.device))
        self.err_d_fake = self.l_bce(self.pred_fake.to(self.device), self.fake_label.to(self.device))

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()

    ##
    def reinit_d(self):
        """ Re-initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('   Reloading net d')

    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        if hasattr(self.opt, 'us_net') and self.opt.us_net:
            if self.opt.cal:
                for w in self.opt.train_width_mult_list[::-1]:
                    self.netg.apply(lambda m: setattr(m, 'width_mult', w))
                    # Forward-pass
                    self.forward_g()
                    self.forward_d()
                
                return
            
            min_width, max_width = min(self.opt.train_width_mult_list), max(self.opt.train_width_mult_list)
            widths_train = []
            for _ in range(self.opt.train_sample_net_num - 2):
                widths_train.append(
                    random.uniform(min_width, max_width))
                
            widths_train = [max_width, min_width] + widths_train
            # logger.info('sampled width: {}'.format(widths_train))

            
            self.optimizer_g.zero_grad()
            self.optimizer_d.zero_grad()
            
            for width_mult in widths_train:
                self.netg.apply(lambda m: setattr(m, 'width_mult', width_mult))
                
                # Forward-pass
                self.forward_g()
                self.forward_d()

                # Backward-pass
                # netg
                self.backward_g()
                # self.optimizer_g.step()

                # netd
                
                self.backward_d()
                # self.optimizer_d.step()
                if self.err_d.item() < 1e-5: self.reinit_d()
                
            self.optimizer_g.step()
            self.optimizer_d.step()
            
            self.netg.apply(lambda m: setattr(m, 'width_mult', 1.0))
                
            return
        
        if hasattr(self.opt, 'fn3_channel') and self.opt.fn3_channel:
            
            fn3_channel_layers_name = [i[0] for i in self.opt.fn3_channel_key_layers_info]
            fn3_channel_channels = [i[1] for i in self.opt.fn3_channel_key_layers_info]
            
            for i, c in enumerate(fn3_channel_channels):
                if fn3_channel_layers_name[i] in self.opt.fn3_channel_disabled_layers:
                    continue
                fn3_channel_channels[i] = random.randint(1, c)
            
            set_fn3_channel_channels(self.netg, fn3_channel_channels)
        
        if hasattr(self.opt, 'ofa') and self.opt.ofa:
            # print('ofa')
            sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/once-for-all')
            from ofa_open_api.ofa_net import sample_ganomaly_ofa_sub_net_width_mult
            sample_ganomaly_ofa_sub_net_width_mult(self.netg, self.opt.train_width_mult_list)
            
        # Forward-pass
        self.forward_g()
        self.forward_d()

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # netd
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d.item() < 1e-5: self.reinit_d()
        
        if hasattr(self.opt, 'fn3_channel') and self.opt.fn3_channel:
            set_fn3_channel_channels(self.netg, [i[1] for i in self.opt.fn3_channel_key_layers_info])

