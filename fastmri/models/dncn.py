import unittest
import merlinth
import torch
from torch._C import device
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from fastmri.layers.fft import fft2c, ifft2c
from fastmri.layers.data_consistency import SingleCoilProxLayer
from fastmri.models.cnn import Real2chCNN

class DnCn(nn.Module):
    def __init__(self, input_dim=1, nc=5, nd=5, nf=64, ks=3, activation='relu'):
        super(DnCn, self).__init__()
        self.nc = nc 
        self.nd = nd
        conv_blocks = []
        dcs = []
        
        for i in range(nc):
            # self.conv_blocks.append(ConvBlock(nd, input_dim, nf, ks, activation))
            conv_blocks.append(Real2chCNN(input_dim=input_dim,filters=nf,num_layer=nd,activation=activation, use_bias=False))
            dcs.append(SingleCoilProxLayer(center_fft=False))
        
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = nn.ModuleList(dcs)

    def forward(self, x, k, m):
        """
        x: input image, [nb, 1， nx, ny], dtype=torch.complex64
        k: sampled kspace, [nb, 2, nx, ny], dtype=torch.complex64
        mask: sampling mask, [nb, 1, nx, ny], dtype=bool
        """
        x = torch.view_as_real(x)
        k = torch.view_as_real(k)
        
        for i in range(self.nc):
            
            x_cnn = self.conv_blocks[i](x)
            x += x_cnn
            x = self.dcs[i](x, k, m)

        return torch.view_as_complex(x)

    
class ConvBlock(nn.Module):
    def __init__(self, nd, input_dim, nf=32, ks=3, activation='relu'):
        super(ConvBlock, self).__init__()
        
        conv_layer = nn.Conv2d
        if activation == 'relu':
            act_layer = nn.ReLU

        conv_first = conv_layer(input_dim, nf, ks, padding='same')
        conv_last = conv_layer(nf, input_dim, ks, padding='same')

        layers = [conv_first]
        for i in range(nd-1):
            layers.append(conv_layer(nf, nf, ks, padding='same'))
            layers.append(act_layer())
        layers.append(conv_last)
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

# class DataConsistancyDnCn(nn.Module):
#     def __init__(self, learn_lambda=False, noise_level=None):
#         super(DataConsistancyDnCn, self).__init__()
#         self.noise_level = noise_level
#         self.learn_lambda = learn_lambda

    
#     def forward(self, x, k0, mask):
#         """
#         x: input image, [nb, nx, ny, 2], dtype=2-channel complex
#         k0: sampled kspace, [nb, nx, ny, 2], dtype=2-channel complex
#         mask: sampling mask, [nb, nx, ny, 1], dtype=bool
#         """

#         x = torch.view_as_complex(x)
#         k0 = torch.view_as_complex(k0)
        
#         k = 

class TestDnCn(unittest.TestCase):
    def testDnCn(self):
        #from merlinth import complex2real
        # from fastmri_dataloader.fastmri_dataloader_th import FastmriCartesianDataset
        x = np.random.randn(4, 2, 320, 320) + 1j * np.random.randn(4, 2, 320, 320)
        k = np.random.randn(4, 2, 320, 320) + 1j * np.random.randn(4, 2, 320, 320)
        mask = np.random.choice(a=[True, False], size=(4, 2, 320, 320))
        
        device = torch.device('cuda')
        x = torch.tensor(x).to(device)
        k = torch.tensor(k).to(device)
        mask = torch.tensor(mask).to(device)
        net = DnCn()


        out = net(x, k, mask)

        print(out)