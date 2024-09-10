import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.fft as fft

from torch.optim import Adam

import torchvision
from torchvision import models

import pytorch_lightning as pl
from pytorch_lightning  import LightningModule

import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pdb
import time

'''
Finished Modules:

'''
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * nn.Sigmoid(x)
        return x

class DownSample(nn.Module):
    def __init__(self,dim):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)
        
    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self,dim):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

'''
Un Finished Modules:

'''

class FFT(nn.Module):
    def __init__(self,reverse = False):#if True means I-FFT
        super(FFT, self).__init__()
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return fft.ifft2(x)
        else:
            return fft.fft2(x)

    
class FSABlock(nn.Module):
    def __init__(self,k):#k for TOPk sparsity
        super(FSABlock, self).__init__()
        self.gnorm = nn.GroupNorm(num_groups,num_channels)
        self.convs = nn.Sequential(
            nn.Conv2d(in_c,out_c,stride = 2,kernel_size = 1),
            nn.Conv2d(in_c,out_c,stride = 2,kernel_size = 3)
        )
        self.FFTQ = FFT()
        self.FFTK = FFT()
        self.IFFTer = FFT(reverse=True)
        self.TopK = xx
        self.Scatter = torch.scatter()
        self.Softmax = nn.softmax()
        self.reshaper = torch.reshape()
        self.finalconv = nn.Conv2d(in_c,out_c,stride = 2,kernel_size = 1)

    def forward(self, x):
        x = self.gnorm(x)
        normx = x
        x = self.convs(x)
        Fv = x
        Fq = self.FFTQ(xxxx)
        Fk = self.FFTK(xxxx)
        F_special = torch.mutal(Fq,Fk)
        A = self.IFFTer(F_special)
        A_bar = self.TopK(A)
        scatter_A_bar = self.Scatter(A_bar)
        Fw = self.Softmax(scatter_A_bar)
        F4 = torch.mutal(Fv,Fw)
        res = self.finalconv(self.reshaper(res))
        res = torch.mutal(normx,res)
        return res

class HighPassFilter(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return x

class LowPassFilter(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return x
    
class FDBlock(nn.Module):
    def __init__(self,edge,skin,normalized_shape):
        super(FDBlock, self).__init__()
        self.edge = edge
        self.skin = skin
        self.common = nn.Sequential(
            nn.LayerNorm(normalized_shape),
            FSABlock(),
            FFT(reverse=False),
        )
        self.edgeer = nn.Sequential(
            HighPassFilter(),
            FFT(reverse=True),
            nn.Conv2d(in_c,out_c,kernel_size = 3),
            nn.ReLU()
        )
        self.skiner = nn.Sequential(
            LowPassFilter(),
            FFT(reverse=True),
            nn.Conv2d(in_c,out_c,kernel_size = 3),
            nn.ReLU()
        )
        
    def forward(self, x):
        assert self.edge ^ self.skin
        if self.edge:
            return self.edgeer(self.common(x))
        if self.skin:
            return self.skiner(self.common(x))
        pass


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_c,out_c,stride = 2,kernel_size = 3)
        self.fsablock1 = FSABlock()
        self.downsample1 = DownSample()
        self.fsablock2 = FSABlock()
        self.downsample2 = DownSample()
        self.fsablock3 = FSABlock()
        self.downsample3 = DownSample()

    def forward(self, x):
        x = self.conv1(x)
        x = self.downsample1(self.fsablock1(x))
        x = self.downsample2(self.fsablock2(x))
        x = self.downsample3(self.fsablock3(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            FSABlock(),
            UpSample())
        self.layer2 = nn.Sequential(
            FSABlock(),
            FSABlock(),
            UpSample())
        self.layer3 = nn.Sequential(
            FSABlock(),
            FSABlock(),
            UpSample())
        self.layer4 = nn.Sequential(
            FSABlock(),
            FSABlock(),
            nn.GroupNorm(num_groups,num_channels),
            Swish(),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_c,out_c,stride = 2,kernel_size = 3)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
