
from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
import glob
import torch
import yaml
from torch.utils.data import DataLoader

import torch.nn as nn
import numpy as np
from torch import optim
from alignment import wagner_fischer, align, naive_backtrace

import json
import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.utils.data as tud
from torch.distributions import Categorical




class ConfidanceModel(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.fc_layer = nn.Sequential(
                                      nn.Linear(1024,512),
                                      nn.ReLU(),
                                      nn.Linear(512,1),
                                      nn.Sigmoid(),
                                      )
        
                                     
    def forward(self,x):
        out = self.fc_layer(x)
        return out
    
    
    

class SpeechDataGenerator:

    def __init__(self, data_folder, batch_size):
        npy_files = sorted(glob.glob(data_folder+'/*.npy'))
        lens = [np.load(filepath, allow_pickle=True).item()['labels'].shape[1] for filepath in npy_files]
        
        bucket_diff = 4
        max_len = max(lens)
        num_buckets = max_len // bucket_diff
        buckets = [[] for _ in range(num_buckets)]
        for d in npy_files:
            bid = min(np.load(d, allow_pickle=True).item()['labels'].shape[1] // bucket_diff, num_buckets - 1)
            buckets[bid].append(d)

        # Sort by input length followed by output length
        sort_fn = lambda x : (round(np.load(d, allow_pickle=True).item()['feats_length'].item(), 1),
                              np.load(d, allow_pickle=True).item()['labels'].shape[1])
        for b in buckets:
            b.sort(key=sort_fn)
        data = [d for b in buckets for d in b]
        self.data = data
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        npy_path = self.data[idx]
        features = np.load(npy_path, allow_pickle=True).item()['embedding'].detach().cpu().numpy()
        target = np.load(npy_path, allow_pickle=True).item()['labels'].detach().cpu().numpy()
        return features, target
        

class BatchRandomSampler(tud.sampler.Sampler):
    """
    Batches the data consecutively and randomly samples
    by batch without replacement.
    """
    
    def __init__(self, data_source, batch_size):
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i + batch_size)
                for i in range(0, it_end, batch_size)]
        self.data_source = data_source
    
    def __iter__(self):
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)
        
    
    


def collate_fun(batch):
    features = []
    labels = []
    max_len = max([item[1].shape[1] for item in batch])
    
    for i in range(len(batch)):
        feats = batch[i][0]
        label = batch[i][1]
        pad = np.zeros((feats.shape[0], max_len-feats.shape[1], feats.shape[2]))
        pad_feats = np.concatenate((feats, pad), axis=1)
        pad = np.ones((label.shape[0], max_len-label.shape[1])).astype('int')
        pad_label = np.concatenate((label, pad),axis=1)
        features.append(pad_feats)
        labels.append(pad_label)
        
    return torch.Tensor(np.concatenate((features))), torch.LongTensor(np.concatenate((labels)))
        
def make_loader(data_folder, preproc,
                batch_size, num_workers=4):
    dataset = SpeechDataGenerator(data_folder,
                           batch_size)
    
    sampler = BatchRandomSampler(dataset, batch_size)
    loader = tud.DataLoader(dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=collate_fun,
                drop_last=True)
    return loader


def normalized_cross_entropy(c,p):
    H_p = torch.mean(Categorical(probs = p).entropy())
    criterion = nn.BCELoss()
    H_c_p =criterion(predictions, labels.float())
    nce = (H_p - H_c_p)/H_p
    return nce
    


     
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='use ctc to generate alignment')
    parser.add_argument('--train_folder', default='./data/aligned_data/train', help='save data folder')
    parser.add_argument('--test_folder', default='.//data/aligned_data/test', help='save data folder')
    parser.add_argument('--epochs', default=100, help='numbder of epochs')
    
    parser.add_argument('--batch_size', type=int, default=25, help='batch size')
    args = parser.parse_args()
    
    train_dataset = SpeechDataGenerator(args.train_folder,
                           args.batch_size)
    
    train_sampler = BatchRandomSampler(train_dataset, args.batch_size)
    train_loader = tud.DataLoader(train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                collate_fn=collate_fun,
                drop_last=True)
    
    #test_dataset = SpeechDataGenerator(args.test_folder,
    #                       args.batch_size)
    #test_sampler = BatchRandomSampler(test_dataset, args.batch_size)
    #train_loader = tud.DataLoader(train_dataset,
    #            batch_size=args.batch_size,
    #            sampler=test_sampler,
    #            collate_fn=collate_fun,
    #            drop_last=True)
    
    use_cuda=True
    device = torch.device('cuda' if use_cuda else 'cpu')
    confidence_model = ConfidanceModel().to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(confidence_model.parameters(), lr=0.001, weight_decay=1e-05, betas=(0.9, 0.98), eps=1e-9)
    
    nce_list = list()
    gt_list = list()
    pred_list = list()
    
    for epoch in range(args.epochs):
        for batch_idx, (feats,labels)  in enumerate(train_loader):
    
            feats = feats.to(device)
            labels = labels.to(device)
            labels = 1-labels
            predictions = confidence_model(feats).squeeze(2)
            loss =criterion(predictions, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nce_val = normalized_cross_entropy(labels, predictions)
            nce_list.append(nce_val.item())
            if batch_idx%500==0:
                print(f'Loss at epoch {epoch} and iteration {batch_idx} is {loss.item()}')
                print(f'NCE Loss at epoch {epoch} and iteration {batch_idx} is {nce_val.item()}')
                
            
            nce_val = normalized_cross_entropy(labels, predictions)
            nce_list.append(nce_val.item())
    print(f'Mean NCE loss {np.mean(nce_list)}')
    
    