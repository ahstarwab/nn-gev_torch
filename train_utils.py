import os
import sys
import time
import numpy as np
import datetime
import pickle as pkl

import torch
import pdb
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import logging
from multiprocessing import Pool

class ModelTrainer:
    def __init__(self, model, train_loader, criterion, optimizer, args, device):

        self.exp_path = os.path.join(args.exp_path, args.tag + '_' + str(np.random.randint(999))) # To avoid overwritten
        
        if not os.path.exists(self.exp_path):
            os.mkdir(self.exp_path)

        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.exp_path, 'training.log'))
        sh = logging.StreamHandler(sys.stdout)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S'))
        sh.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        self.logger.info(f'Current Exp Path: {self.exp_path}')

        self.writter = SummaryWriter(os.path.join(self.exp_path, 'logs'))

        self.model_type = args.model_type
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.criterion = criterion

        self.optimizer = optimizer

        if args.load_ckpt:
            self.load_checkpoint(args.load_ckpt)


    def train(self, num_epochs):
        self.logger.info('Model Type: '+str(self.model_type))
        self.logger.info('TRAINING .....')

        for epoch in tqdm(range(0, num_epochs)):
            self.logger.info("==========================================================================================")

            train_loss = self.train_single_epoch()
            valid_loss = self.inference()

            # For longer message, it would be better to use this kind of thing.
            logging_msg1 = (
                f'| Epoch: {epoch:02} | Train Loss: {train_loss:0.6f} '
            )

            logging_msg2 = (
                f'| Epoch: {epoch:02} | Valid Loss: {valid_loss:0.6f} '
            )

            self.logger.info("------------------------------------------------------------------------------------------")
            self.logger.info(logging_msg1)
            self.logger.info(logging_msg2)

            self.save_checkpoint(epoch, train_loss)

            # Log values to Tensorboard
            self.writter.add_scalar('data/Train_Loss', train_loss, epoch)


        self.writter.close()
        self.logger.info("Training Complete! ")
        


    def train_single_epoch(self):
        """Trains the model for a single round."""
        self.model.train()
        total_loss = 0.0
        
        for b, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            IBM_X, IBM_N, Y_abs, IBM_X_len, _ = batch

            IBM_X = IBM_X.to(self.device)
            IBM_N = IBM_N.to(self.device)
            Y_abs = Y_abs.to(self.device)
            IBM_X_len = IBM_X_len.to(self.device)

            batch_size = len(IBM_X_len)

            st_pt = 0
            loss_X, loss_N = 0, 0
            for end_pt in IBM_X_len:
                
                X_mask_hat, N_mask_hat = self.model(Y_abs[st_pt:st_pt+end_pt])

                X_mask_hat = X_mask_hat.reshape(-1, 6, 513)
                N_mask_hat = N_mask_hat.reshape(-1, 6, 513)
                
                loss_X += self.criterion(X_mask_hat, IBM_X[st_pt:st_pt+end_pt]).mean()
                loss_N += self.criterion(N_mask_hat, IBM_N[st_pt:st_pt+end_pt]).mean()
                
                st_pt += end_pt

            batch_loss = (loss_X + loss_N) / 2
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item() * batch_size
    
        return total_loss

    def inference(self):
        self.model.eval()  # Set model to evaluate mode.
        total_loss = 0.0
    
        with torch.no_grad():
            for b, batch in enumerate(self.train_loader):
                
                IBM_X, IBM_N, Y_abs, IBM_X_len, _ = batch

                IBM_X = IBM_X.to(self.device)
                IBM_N = IBM_N.to(self.device)
                Y_abs = Y_abs.to(self.device)
                IBM_X_len = IBM_X_len.to(self.device)

                batch_size = len(IBM_X_len)

                st_pt = 0
                loss_X, loss_N = 0, 0
                for end_pt in IBM_X_len:
                    
                    X_mask_hat, N_mask_hat = self.model(Y_abs[st_pt:st_pt+end_pt])

                    X_mask_hat = X_mask_hat.reshape(-1, 6, 513)
                    N_mask_hat = N_mask_hat.reshape(-1, 6, 513)
                    
                    loss_X += self.criterion(X_mask_hat, IBM_X[st_pt:st_pt+end_pt]).mean()
                    loss_N += self.criterion(N_mask_hat, IBM_N[st_pt:st_pt+end_pt]).mean()
                    
                    st_pt += end_pt
                    

                batch_loss = (loss_X + loss_N) / 2
                total_loss += batch_loss.item() * batch_size

        return total_loss

    def get_lr(self):
        """Returns Learning Rate of the Optimizer."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def save_checkpoint(self, epoch, loss):

        state_dict = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learning_rate': self.get_lr(),
            'exp_path': self.exp_path,
            'loss': loss,
        }
        save_path = "{}/ck_{}_{:0.4f}.pth.tar".format(self.exp_path, epoch, loss)
        torch.save(state_dict, save_path)

    def load_checkpoint(self, ckpt):
        self.logger.info(f"Loading checkpoint from {ckpt}")
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state'], strict=False)
