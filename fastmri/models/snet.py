import torch
from torch import nn
import unittest
import merlinth
import numpy as np
import torch.nn.functional as F
from fastmri.layers.fft import fft2, ifft2, fft2c, ifft2c

class CNNLayer(nn.Module):
    def __init__(self, n_f=32):
        super(CNNLayer, self).__init__()
        layers = [
            nn.Conv2d(in_channels=2, out_channels=n_f, kernel_size=3, padding='same', bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=n_f, out_channels=n_f, kernel_size=3, padding='same', bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=n_f, out_channels=2, kernel_size=3, padding='same', bias=False),
        ]
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class SCell(nn.Module):
    def __init__(self, n_f, learn_step=True, weight_init=1, center_fft=False, is_last=False):
        super(SCell, self).__init__()
        self.conv_block = CNNLayer(n_f=n_f)
        self.learn_step = learn_step
        self.is_last = is_last
        if learn_step:
            self._gamma = torch.nn.Parameter(torch.Tensor(1))
            self._gamma.data = torch.tensor(weight_init, dtype=self._gamma.dtype)
        
        self.fft = fft2c if center_fft else fft2
        self.ifft = ifft2c if center_fft else ifft2

    def forward(self, x, k0, mask):
        x = merlinth.complex2real(x)
        s = self.conv_block(x)
        s = merlinth.real2complex(s)
        dc = self.dataconsis(s, k0, mask)
        if not self.is_last:
            gamma = F.relu(self._gamma)
        else:
            gamma = 1
        x = s - gamma * dc
    
        return x, k0, mask
    
    def dataconsis(self, s, k0, mask):
        resk = self.fft(s) * mask - k0
        return self.ifft(resk)
        


class Snet(nn.Module):
    def __init__(self, n_blcoks=10, input_dim=2, n_f=64, activation='lrelu', learn_step=True):
        super(Snet, self).__init__()
        self.n_blocks = n_blcoks
        layers = [SCell(n_f=n_f, learn_step=learn_step) for i in range(n_blcoks-1)]
        layers.append(SCell(n_f=n_f, learn_step=learn_step, is_last=True))
        #self.seq = nn.Sequential(*layers)
        self.layers = nn.ModuleList(layers)

    def forward(self, x, k, m, *args):
        #x, _, _ = self.seq(x, k, m)
        for i in range(self.n_blocks):
            x, _, _ = self.layers[i](x, k, m)
        return x
        

    

