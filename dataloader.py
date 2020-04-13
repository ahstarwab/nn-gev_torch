import os
import pickle
import multiprocessing as mp
import json
import cv2
import numpy as np
from PIL import Image
import tqdm
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from utils.mask_utils import estimate_IBM
from utils.signal_utils import stft, audioread

from pathlib import Path
import pdb

class ParallelSim(object):
    def __init__(self, processes):
        self.pool = mp.Pool(processes=processes)
        self.total_processes = 0
        self.completed_processes = 0
        
    def add(self, func, args):
        self.pool.apply_async(func=func, args=args, callback=self.complete)
        self.total_processes += 1

    def complete(self, result_tuple):
        self.IBM_X, self.IBM_N, self.Y_abs = result_tuple

        self.completed_processes += 1
        print('-- processed {:d}/{:d}'.format(self.completed_processes,
                                              self.total_processes), end='\r')

    def run(self):
        self.pool.close()
        self.pool.join()

    def get_results(self):
        return self.IBM_X, self.IBM_N, self.Y_abs



def Chime_Collate(batch):

    IBM_X, IBM_N, Y_abs, fname = list(zip(*batch))

    IBM_X_len = torch.LongTensor(np.array([len(x) for x in IBM_X]))

    IBM_X = torch.FloatTensor(np.concatenate(IBM_X, axis=0))
    IBM_N = torch.FloatTensor(np.concatenate(IBM_N, axis=0))
    Y_abs = torch.FloatTensor(np.concatenate(Y_abs, axis=0))

    data = (IBM_X, IBM_N, Y_abs, IBM_X_len, fname)

    return data


class Chime_Dataset(Dataset):
    def __init__(self, data_partition, args, num_workers=None):

        super(Chime_Dataset, self).__init__()


        if args.write_cache: # Default : False
            if num_workers:
                self.num_workers = num_workers
            else:
                self.num_workers = mp.cpu_count()

            num_processes = mp.cpu_count()
            runner = ParallelSim(processes=num_processes)
            runner.add(self.dump_data_cache, (args.data_dir, args.cache_dir))
            runner.run()
            # results = runner.get_results()

        dir_ = Path(args.cache_dir)
        with open(dir_.joinpath('flist_{}.json'.format(data_partition))) as fid:
            data_list = json.load(fid)

        self.data_list = data_list[:100]
        
        num_processes = mp.cpu_count()
        runner = ParallelSim(processes=num_processes)
        runner.add(self.load_cache, (self.data_list, args.cache_dir))
        runner.run()
        self.IBM_X, self.IBM_N, self.Y_abs = runner.get_results()
        
        #  = self.load_cache(args.cache_dir, data_partition)

    def __getitem__(self, idx):
        IBM_X = self.IBM_X[idx]
        IBM_N = self.IBM_N[idx]
        Y_abs = self.Y_abs[idx]
        fname = self.data_list[idx]
        return IBM_X, IBM_N, Y_abs, fname

    def __len__(self):
        return len(self.IBM_X)
    
    def gen_flist_simu(self, data_dir, stage, ext=False):

        
        with open(os.path.join(data_dir, 'data', 'annotations', '{}05_{}.json'.format(stage, 'simu'))) as fid:
            annotations = json.load(fid)
        if ext:
            isolated_dir = 'isolated_ext'
        else:
            isolated_dir = 'isolated'
        flist = [os.path.join(data_dir, 'data', 'audio', '16kHz', isolated_dir,
                '{}05_{}_{}'.format(stage, a['environment'].lower(), 'simu'),
                '{}_{}_{}'.format(a['speaker'], a['wsj_name'], a['environment']))
                for a in annotations]
        return flist


    def gen_flist_real(self, data_dir, stage):
        with open(os.path.join(data_dir, 'data', 'annotations','{}05_{}.json'.format(stage, 'real'))) as fid:
            annotations = json.load(fid)
        flist_tuples = [(os.path.join(
                data_dir, 'data', 'audio', '16kHz', 'embedded', a['wavfile']),
                        a['start'], a['end'], a['wsj_name']) for a in annotations]
        return flist_tuples


    def get_audio_data(self, file_template, postfix='', ch_range=range(1, 7)):
        audio_data = list()
        for ch in ch_range:
            audio_data.append(audioread(
                    file_template + '.CH{}{}.wav'.format(ch, postfix))[None, :])
        audio_data = np.concatenate(audio_data, axis=0)
        audio_data = audio_data.astype(np.float32)
        return audio_data


    #TODO 
    #MultiProcessing
    def dump_data_cache(self, chime_data_dir, dest_dir):

        dest_path = Path(dest_dir)
        for stage in ['tr', 'dt']:
            
            flist = self.gen_flist_simu(chime_data_dir, stage, ext=True)
            export_flist = list()
            
            dest_path.joinpath(stage).mkdir(parents=True, exist_ok=True)

            for f in tqdm.tqdm(flist, desc='Generating data for {}'.format(stage)):
                clean_audio = self.get_audio_data(f, '.Clean')
                noise_audio = self.get_audio_data(f, '.Noise')

                X = stft(clean_audio, time_dim=1).transpose((1, 0, 2))
                N = stft(noise_audio, time_dim=1).transpose((1, 0, 2))
                IBM_X, IBM_N = estimate_IBM(X, N)
                            
                Y_abs = np.abs(X + N)
                export_dict = {
                    'IBM_X': IBM_X.astype(np.float32),
                    'IBM_N': IBM_N.astype(np.float32),
                    'Y_abs': Y_abs.astype(np.float32)
                }
                export_name = os.path.join(dest_dir, stage, f.split('/')[-1]+'.pkl')

                #Dump data
                with open(export_name, 'wb') as fid:
                    pickle.dump(export_dict, fid)

                export_flist.append(os.path.join(stage, f.split('/')[-1]))
                
            #Dump file_lists
            with open(os.path.join(dest_dir, 'flist_{}.json'.format(stage)),
                    'w') as fid:
                json.dump(export_flist, fid, indent=4)


    
    def load_cache(self, data_list, cache_dir):
        IBM_X, IBM_N, Y_abs = [], [], []
        dir_ = Path(cache_dir)
        for fname in tqdm.tqdm(data_list):
            with open(dir_.joinpath(fname+'.pkl'), 'rb') as f:
                data_ = pickle.load(f)
                IBM_X.append(data_['IBM_X'])
                IBM_N.append(data_['IBM_N'])
                Y_abs.append(data_['Y_abs'])

        return IBM_X, IBM_N, Y_abs