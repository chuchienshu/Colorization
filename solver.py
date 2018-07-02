from model import ColorizationNet
from torch.autograd import Variable
from losses import CE_loss
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime

import torch.optim as optim
import shutil
import numpy as np
from utils import tools

gamut = np.load('models/custom_layers/pts_in_hull.npy')

class Solver(object):

    def __init__(self, config):
        """Initialize configurations."""

        self.model = ColorizationNet(config['bachnorm'], config['pretrained'])
        self.criterion = CE_loss()
        self.lr = config['lr']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-3, betas=(0.8, 0.9))
        self.record_iters = 0
        self.resume_iters = None
        self.test_cycle = config['test_cycle']

        self.cuda = config['cuda']
        self.num_iters = config['num_iters']
        self.lr_update_step = config['lr_update_step']
        self.log_step = config['log_frequency']

        # Directories.
        self.model_save_dir = config['save']
        self.lr_update_step = config['lr_update_step']
        self.result_dir = config['save']

        if self.cuda:
            self.model.cuda()


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step %d ...' % (resume_iters))
        model_path = os.path.join(self.model_save_dir, '%d_checkpoint.pth.tar' % (resume_iters))

        checkpoint = torch.load(model_path)
        start_iters = checkpoint['iters']
        # G_best_err = G_checkpoint['best_err']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.lr = checkpoint['lr']

        return start_iters + 1

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train(self, train_data_loader,test_data_loader, train_logger, test_logger,resume_iters=0,valitate=True ):

        data_iter = iter(train_data_loader)

        # Start training from scratch or resume training.
        start_iters = 0
        if resume_iters:
            start_iters = self.restore_model(resume_iters)

        # Start training.
        print('Start training...')
        since = time.time()
        
        self.model.train()  # Set g_model to training mode
        
        for global_iteration in range(start_iters, self.num_iters):
            try:
                gt  = next(data_iter)
            except:
                data_iter = iter(train_data_loader)
                gt  = next(data_iter)

            '''
            out = torchvision.utils.make_grid(torch.cat([ gt, frame_1], dim = 0),nrow= 8, pad_value=1, padding=6)
            tools.img_show(out)         
            exit() 
            '''
            # wrap them in Variable
            if self.cuda:
                gt = Variable(gt.cuda())
            else:
                gt = Variable(gt) 

            ###################### Train discrimlator ######################################

            wei_output, enc_gt = self.model(gt)


            loss = self.criterion(wei_output ,enc_gt)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #################### LOGGING #############################
            if global_iteration % self.log_step == 0:
                lr = self.optimizer.param_groups[0]['lr']
                # train_logger.add_image('img', img.data, global_iteration)
                # train_logger.add_image('gt', gt.data, global_iteration)
                train_logger.add_scalar('lr',  lr, global_iteration)
                train_logger.add_scalar('loss',  loss , global_iteration)
        
            print('training %d iters,loss is %.4f' % ( global_iteration, loss))
            # Decay learning rates.
            if (global_iteration+1) % self.lr_update_step == 0:
                self.record_iters = global_iteration
                if self.lr > 1e-8:
                    self.lr *= 0.316
                self.update_lr(self.lr )
                if valitate and (global_iteration+1) % (self.lr_update_step * self.test_cycle) == 0:
                    self.test(test_data_loader, test_logger)
                    self.model.train()  # Set g_model to training mode back
                print ('Decayed learning rates, lr: %4f' % (self.lr ))
            
        time_elapsed = time.time() - since
        print('train completed in %.0fm %.0fs'% (time_elapsed // 60, time_elapsed % 60))
    
    def save_checkpoint(self,state,  path, prefix,iters, filename='checkpoint.pth.tar'):
        prefix_save = os.path.join(path, prefix)
        name = '%s_%d_%s' % (prefix_save,iters,filename)
        torch.save(state, name)
        shutil.copyfile(name,  '%s_latest.pth.tar' % (prefix_save))


    def test(self, data_loader,test_logger,inference_iter=0, save_model = True):
        # Load the trained generator.
        self.optimizer.zero_grad()
        data_iter = iter(data_loader)
        self.model.nnecnclayer.nnenc.alreadyUsed = False

        if inference_iter:
            self.restore_model(inference_iter)

        if inference_iter:
            print('Start inferencing...')
        else:
            print('Start testing...')
        since = time.time()
        
        self.model.eval()  # Set g_model to training mode

        img_dir =  '%simg/%d/' % (self.result_dir, self.record_iters)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        len_record = len(data_loader)
        softmax_op = torch.nn.Softmax()
        test_loss = 0.0

        for global_iteration in range(len_record):
            # Each epoch has a training and validation phase
            print('completed %d of %d' % (global_iteration, len_record))
            # Iterate over data.
            gt = next(data_iter)

            '''
            out = torchvision.utils.make_grid(torch.cat([ gt, frame_1], dim = 0),nrow= 8, pad_value=1, padding=6)
            tools.img_show(out)         
            exit() 
            '''
            # wrap them in Variable
            if self.cuda:
                gt = Variable(gt.cuda(), volatile=True)
            else:
                gt = Variable(gt, volatile=True) 

            full_rs_output, wei_output, enc_gt = self.model(gt)
            loss = self.criterion(wei_output ,enc_gt)
            test_loss += loss.data[0]

            gt_img_l = gt[:,:1,:,:]
            # _, _, H_orig, W_orig = gt_img_l.data.shape

            # post-process
            full_rs_output *= 2.606
            full_rs_output = softmax_op(full_rs_output).cpu().data.numpy()

            fac_a = gamut[:,0][np.newaxis,:,np.newaxis,np.newaxis]
            fac_b = gamut[:,1][np.newaxis,:,np.newaxis,np.newaxis]

            img_l = gt_img_l.cpu().data.numpy().transpose(0,2,3,1)
            frs_pred_ab = np.concatenate((np.sum(full_rs_output * fac_a, axis=1, keepdims=True), np.sum(full_rs_output * fac_b, axis=1, keepdims=True)), axis=1).transpose(0,2,3,1)
            
            frs_predic_imgs = np.concatenate((img_l, frs_pred_ab ), axis = 3)
            tools.save_imgs(frs_predic_imgs, '%s%d_frspredic_' %  (img_dir, global_iteration))
            gt = gt.cpu().data.numpy().transpose(0,2,3,1).astype('float64')
            tools.save_imgs(gt,'%s%d_gt_' %  (img_dir ,global_iteration))

        best_error = test_loss / len_record

        if save_model:

            self.save_checkpoint({'arch':'ColorizationNet',
                                'lr':self.lr,
                                'iters':self.record_iters,
                                'state_dict': self.model.state_dict(),
                                'error':best_error},
                                self.model_save_dir,'G_', self.record_iters)

        test_logger.add_scalar('test_lr',  self.lr, self.record_iters)
        test_logger.add_scalar('test_loss',  best_error, self.record_iters)
        self.model.nnecnclayer.nnenc.alreadyUsed = False
        time_elapsed = time.time() - since
        print('test loss is %.4f' % (best_error ))
        print('test completed in %.0fm %.0fs'% (time_elapsed // 60, time_elapsed % 60))