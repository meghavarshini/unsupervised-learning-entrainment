import os
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
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
#-------------------------------------------------
# Define the dataset class
#-------------------------------------------------
class EntDataset(Dataset):
    def __init__(self, file_name):
        self.file_name = file_name
        
    def __getitem__(self, idx):
        hf = h5py.File(self.file_name, 'r')
        X = hf['dataset'][idx,0:228]
        Y = hf['dataset'][idx,228:456]
        hf.close()
        return (X, Y)

    def __len__(self):
        hf = h5py.File(self.file_name, 'r')
        length=hf['dataset'].shape[0]
        hf.close()
        return length




class LSTM_Entrainment(nn.Module):
    def __init__(self, n_params):
        # input is same as input- so n_params will be same
        #  since we aren't doing classificaiton task, params depends on no. of features
        super(LSTM_Entrainment, self).__init__()
        # the model is run over
        zdim=50
        self.lstm = nn.LSTM(input_size=n_params, hidden_size=zdim, num_layers=1, batch_first=True)
        self.ln = nn.Linear(zdim, n_params)


    def forward(self, x):
        # Forward propagate the LSTM
        out, _ = self.lstm(x)
        
        # Reshape output to fit the linear layer:
        # out: batch_size, sequence_length, hidden_dim
        out = out.contiguous().view(-1, self.hidden_dim)

        # Pass the output of each time step through a linear layer; expect logits for each class
        out = self.ln(out)
        
        # Reshape again to get output for each time step
        # Preserves batch size. Second dim is receiving the hidden dimension
        out = out.view(x.size(0), -1, out.size(-1))
        
        return out


def loss_function(pred, true_val):
    loss_fn = nn.MSELoss()
    loss = loss_fn(pred, true_val) #vector of losses

    return loss