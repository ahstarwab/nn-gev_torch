import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('agg')
import pickle as pkl
import torch
import pdb

import cv2
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from multiprocessing import Pool
from pathlib import Path


class ModelTest:
    # Codes were matched based on the ModelTrainer argument order
    def __init__(self, model, data_loader, criterion, args, device):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.out_dir = args.test_dir
        self.render = args.test_render
        self.criterion = criterion
                
        self.load_checkpoint(args.test_ckpt)


    def load_checkpoint(self, ckpt):
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state'], strict=False)

            
    def test(self):
        print('Starting model test.....')
        self.model.eval()  # Set model to evaluate mode.
        total_loss = 0.0
        
        with torch.no_grad():
            for b, batch in enumerate(self.data_loader):
                
                IBM_X, IBM_N, Y_abs, IBM_X_len, fname = batch

                IBM_X = IBM_X.to(self.device)
                IBM_N = IBM_N.to(self.device)
                Y_abs = Y_abs.to(self.device)
                IBM_X_len = IBM_X_len.to(self.device)

                batch_size = len(IBM_X_len)

                st_pt = 0
                loss_X, loss_N = 0, 0
                
                for idx, end_pt in enumerate(IBM_X_len):
                    
                    X_mask_hat, N_mask_hat = self.model(Y_abs[st_pt:st_pt+end_pt])

                    X_mask_hat = X_mask_hat.reshape(-1, 6, 513)
                    N_mask_hat = N_mask_hat.reshape(-1, 6, 513)
                    
                    loss_X += self.criterion(X_mask_hat, IBM_X[st_pt:st_pt+end_pt]).mean()
                    loss_N += self.criterion(N_mask_hat, IBM_N[st_pt:st_pt+end_pt]).mean()

                    if self.render:
                        self.render_mask(X_mask_hat, N_mask_hat, IBM_X[st_pt:st_pt+end_pt], IBM_N[st_pt:st_pt+end_pt], Y_abs[st_pt:st_pt+end_pt], fname[idx])
                    st_pt += end_pt

                batch_loss = (loss_X + loss_N) / 2
                total_loss += batch_loss.item() * batch_size

        return total_loss
        # if self.render:
        #     pool.apply(self.write_img_output, (candidate_i, src_traj_i, src_lens_i, tgt_traj_i, tgt_lens_i, map_file_i, output_file_i))


    def render_mask(self, X_mask_hat, N_mask_hat, IBM_X, IBM_N, Y_abs, fname):

        X_mask_hat = X_mask_hat[:,0,:].cpu()
        N_mask_hat = N_mask_hat[:,0,:].cpu()
        IBM_X = IBM_X[:,0,:].cpu()    
        IBM_N = IBM_N[:,0,:].cpu()
        Y_abs = Y_abs[:,0,:].cpu()


        out_dir = Path(self.out_dir).joinpath(fname)
        out_dir.mkdir(parents=True, exist_ok=True)

        
        fig = plt.figure(figsize=(50, 50))
        ax = plt.axes()
        ax.imshow(X_mask_hat, cmap='viridis')
        plt.savefig(out_dir.joinpath('X_hat.png'))
        plt.close(fig)

        fig = plt.figure(figsize=(50, 50))
        ax = plt.axes()
        ax.imshow(N_mask_hat,  cmap='viridis')
        plt.savefig(out_dir.joinpath('N_hat.png'))
        plt.close(fig)

        fig = plt.figure(figsize=(50, 50))
        ax = plt.axes()
        ax.imshow(IBM_X,  cmap='viridis')
        plt.savefig(out_dir.joinpath('IBM_X.png'))
        plt.close(fig)


        fig = plt.figure(figsize=(50, 50))
        ax = plt.axes()
        ax.imshow(IBM_N,  cmap='viridis')
        plt.savefig(out_dir.joinpath('IBM_N.png'))
        plt.close(fig)

        fig = plt.figure(figsize=(50, 50))
        ax = plt.axes()
        ax.imshow(Y_abs,  cmap='viridis')
        plt.savefig(out_dir.joinpath('Y_abs.png'))
        plt.close(fig)


        print(fname)
        # # self.out_dir
        # fig = plt.imshow(data)
        # fig = plt.figure(figsize=(float(H) / float(80), float(W) / float(80)))
        # # cv2.imwrite(output_file, buffer)
        # ax = plt.axes()
        # ax.clear()
        # plt.close(fig)




    # @staticmethod
    # def write_img_output(gen_trajs, src_trajs, src_lens, tgt_trajs, tgt_lens, map_file, output_file):
    #     """abcd"""
    #     if '.png' in map_file:
    #         map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)
    #         map_array = cv2.cvtColor(map_array, cv2.COLOR_BGR2RGB)

    #     elif '.pkl' in map_file:
    #         with open(map_file, 'rb') as pnt:
    #             map_array = pkl.load(pnt)

    #     # with open(map_file, 'rb') as f:
    #     #     map_array = pkl.load(f)
        
    #     H, W = map_array.shape[:2]
    #     fig = plt.figure(figsize=(float(H) / float(80), float(W) / float(80)),
    #                     facecolor='k', dpi=80)

    #     ax = plt.axes()
    #     ax.imshow(map_array, extent=[-56, 56, 56, -56])
    #     ax.set_aspect('equal')
    #     ax.set_xlim([-56, 56])
    #     ax.set_ylim([-56, 56])

    #     plt.gca().invert_yaxis()
    #     plt.gca().set_axis_off()
    #     plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
    #                         hspace = 0, wspace = 0)
    #     plt.margins(0,0)
        
    #     num_tgt_agents, num_candidates = gen_trajs.shape[:2]
    #     num_src_agents = len(src_trajs)

    #     for k in range(num_candidates):
    #         gen_trajs_k = gen_trajs[:, k]

    #         x_pts_k = []
    #         y_pts_k = []
    #         for i in range(num_tgt_agents):
    #             gen_traj_ki = gen_trajs_k[i]
    #             tgt_len_i = tgt_lens[i]
    #             x_pts_k.extend(gen_traj_ki[:tgt_len_i, 0])
    #             y_pts_k.extend(gen_traj_ki[:tgt_len_i, 1])

    #         ax.scatter(x_pts_k, y_pts_k, s=0.5, marker='o', c='b')
        
    #     x_pts = []
    #     y_pts = []
    #     for i in range(num_src_agents):
    #             src_traj_i = src_trajs[i]
    #             src_len_i = src_lens[i]
    #             x_pts.extend(src_traj_i[:src_len_i, 0])
    #             y_pts.extend(src_traj_i[:src_len_i, 1])

    #     ax.scatter(x_pts, y_pts, s=2.0, marker='x', c='g')

    #     x_pts = []
    #     y_pts = []
    #     for i in range(num_tgt_agents):
    #             tgt_traj_i = tgt_trajs[i]
    #             tgt_len_i = tgt_lens[i]
    #             x_pts.extend(tgt_traj_i[:tgt_len_i, 0])
    #             y_pts.extend(tgt_traj_i[:tgt_len_i, 1])

    #     ax.scatter(x_pts, y_pts, s=2.0, marker='o', c='r')

    #     fig.canvas.draw()
    #     buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #     buffer = buffer.reshape((H, W, 3))

    #     buffer = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(output_file, buffer)
    #     ax.clear()
    #     plt.close(fig)




