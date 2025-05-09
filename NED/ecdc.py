import os
import sys
import h5py
import pdb
import numpy as np 
import csv
import argparse
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt



#-------------------------------------------------
# Define the dataset class
#-------------------------------------------------
class EntDataset(Dataset):
    def __init__(self, file_name, y_mean, y_std):
        self.file_name = file_name
        self.y_mean = y_mean  # y_mean calculated from training data
        self.y_std = y_std    # y_std calculated from training data

        # Open HDF5 file to compute Y mean and std
        with h5py.File(self.file_name, 'r') as hf:
            data = hf['dataset'][:, featDim:2*featDim]  # all Y samples
            self.y_mean = np.mean(data, axis=0)
            self.y_std = np.std(data, axis=0)
    def __getitem__(self, idx):
        with h5py.File(self.file_name, 'r') as hf:
            X = hf['dataset'][idx, 0:featDim]
            Y = hf['dataset'][idx, featDim:2*featDim]
        
        # Normalize Y
        Y_norm = (Y - self.y_mean) / (self.y_std + 1e-8)  # avoid divide-by-zero
        return (X, Y_norm)
    def __len__(self):
        with h5py.File(self.file_name, 'r') as hf:
            return hf['dataset'].shape[0]

# class EntDataset(Dataset):
#     def __init__(self, file_name):
#         self.file_name = file_name
        
#     def __getitem__(self, idx):
#         hf = h5py.File(self.file_name, 'r')
#         X = hf['dataset'][idx,0:featDim]
#         Y = hf['dataset'][idx,featDim:2*featDim]
#         hf.close()
#         return (X, Y)

#     def __len__(self):
#         hf = h5py.File(self.file_name, 'r')
#         length=hf['dataset'].shape[0]
#         hf.close()
#         return length


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Tensor with many dimensions to one with 30- see Pg 5 of the paper
        zdim=30
        intDim = 128                                 # CHANGE zdim here
        self.fc1 = nn.Linear(featDim, intDim)
        self.bn1 = nn.BatchNorm1d(intDim)
        self.fc2 = nn.Linear(intDim, zdim)
        self.bn2 = nn.BatchNorm1d(zdim)

        self.fc3 = nn.Linear(zdim, intDim)
        self.bn3 = nn.BatchNorm1d(intDim)
        self.fc4 = nn.Linear(intDim, featDim)
        self.bn4 = nn.BatchNorm1d(featDim)

        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()


    def encode(self, x):
        h1 = self.fc1(x)
        h1= self.relu(self.bn1(h1))

        h2 = self.fc2(h1)
        z = self.bn2(h2)  # can experiment with this

        return z

    # def reparameterize(self, mu, logvar):
    #     if self.training:
    #         std = logvar.mul(0.5).exp_()
    #         eps = Variable(std.data.new(std.size()).normal_())
    #         return eps.mul(std).add_(mu)
    #     else:
    #         return mu

    def decode(self, z):
        h3 = self.fc3(z)
        h3 = self.bn3(h3)
        h3 = self.relu(h3)
        h4 = self.fc4(h3)
        # h4 = self.sigmoid(h4)
        h4 = self.bn4(h4)
        return h4

    def forward(self, x):
        z = self.encode(x.view(-1, featDim))
        # z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def embedding(self, x):
        z = self.encode(x.view(-1, featDim))
        return z


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x1, x2):
    if loss=='l1':
        BCE = F.smooth_l1_loss(recon_x1, x2.view(-1, featDim), reduction='sum') #option: reduction='mean'
    elif loss=='l2':
        BCE = F.mse_loss(recon_x1, x2.view(-1, featDim), size_average=False)
    

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE #+ KLD #, BCE, KLD


def lp_distance(x1, x2, p):
    dist = torch.dist(x1, x2,p)
    return dist

# Parameters

featDim =228
zdim=30                     # CHANGE zdim here
intDim = 64                                
loss = 'l1'
