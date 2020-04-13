import argparse
import json

import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from chime_data import prepare_training_data
from fgnt.utils import Timer
from fgnt.utils import mkdir_p

from nn_models import BLSTMMaskEstimator
from nn_models import SimpleFWMaskEstimator

from dataloader import Chime_Dataset, Chime_Collate

import pdb

def train(args):
    from train_utils import ModelTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare Data
    train_dataset = Chime_Dataset('tr', args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, 
                            collate_fn=lambda x: Chime_Collate(x), num_workers=args.num_workers)

    # Prepare model
    if args.model_type == 'BLSTM':
        model = BLSTMMaskEstimator()
        model_save_dir = os.path.join(args.data_dir, 'BLSTM_model')
        mkdir_p(model_save_dir)
    elif args.model_type == 'FW':
        model = SimpleFWMaskEstimator()
        model_save_dir = os.path.join(args.data_dir, 'FW_model')
        mkdir_p(model_save_dir)
    else:
        raise ValueError('Unknown model type. Possible are "BLSTM" and "FW"')

    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    trainer = ModelTrainer(model, train_loader, criterion, optimizer, args, device)
    trainer.train(args.num_epochs)

def test(args):
    from test_utils import ModelTest

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare Data
    test_dataset = Chime_Dataset('dt', args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, 
                            collate_fn=lambda x: Chime_Collate(x), num_workers=args.num_workers)

    # Prepare model
    if args.model_type == 'BLSTM':
        model = BLSTMMaskEstimator()
        model_save_dir = os.path.join(args.data_dir, 'BLSTM_model')
        mkdir_p(model_save_dir)
    elif args.model_type == 'FW':
        model = SimpleFWMaskEstimator()
        model_save_dir = os.path.join(args.data_dir, 'FW_model')
        mkdir_p(model_save_dir)
    else:
        raise ValueError('Unknown model type. Possible are "BLSTM" and "FW"')

    
    criterion = torch.nn.BCELoss()
    
    tester = ModelTest(model, test_loader, criterion, args, device)
    tester.test()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model_type',
                        help='Type of model (BLSTM or FW)')
    
    parser.add_argument('--exp_path', default=None, help='Experiment Path')
    parser.add_argument('--tag', default='debugging')
    parser.add_argument('--load_ckpt', default=None, help='Load Model Checkpoint')

    # Data
    parser.add_argument('--cache_dir', default='')
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--write_cache', default=False)


    # Hardware Parameters
    parser.add_argument('--gpu_device', type=str, default='0', help="GPU IDs for model running")
    parser.add_argument('--num_workers', type=int, default=25, help="")

    # Optimization Parameters
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_epochs', default=25, type=int)
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate')       
    parser.add_argument('--dropout', default=0.5, type=float)
    
    # Test Parameters
    parser.add_argument('--test_ckpt', default=None, help="Model Checkpoint for test")
    parser.add_argument('--test_dir', default='./mask_figures', help="Model Checkpoint for test")
    parser.add_argument('--test_render', default=False, help="Model Checkpoint for test")



    # parser.add_argument('--patience', default=5, type=int,
    #                     help='Max. number of epochs to wait for better CV loss')


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    if args.test_ckpt is None: 
        train(args)
    else:
        test(args)