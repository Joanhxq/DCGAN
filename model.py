# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:34:01 2019

@author: hxq
"""

from torch import nn



class NetG(nn.Module):
    def __init__(self, opt):
        super(NetG, self).__init__()
        nz = opt.nz
        ngf = opt.ngf
        
        self.netg = nn.Sequential(
                # the input size is 100 x 1 x 1
                nn.ConvTranspose2d(nz, ngf*8, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm2d(ngf*8),
                nn.ReLU(True),
                # the output size is 1024 x 4 x 4
                nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1),
                nn.BatchNorm2d(ngf*4),
                nn.ReLU(True),
                # the output size is 512 x 8 x 8
                nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1),
                nn.BatchNorm2d(ngf*2),
                nn.ReLU(True),
                # the output size is 256 x 16 x16
                nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # the output size is 128 x 32 x 32
                nn.ConvTranspose2d(ngf, 3, 4, 2, 1),
                nn.Tanh()
                # the output size is 3 x 64 x 64
                )
        
    def forward(self, input):
        x = self.netg(input)
        return x

class NetD(nn.Module):  # Discriminator Define
    def __init__(self, opt):
        super(NetD, self).__init__()
        
        ndf = opt.ndf
        self.netd = nn.Sequential(
                # the input size is 3 x 64 x 64
                nn.ReflectionPad2d(1),
                nn.Conv2d(3, ndf, 4, 2),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # the output size is 128 x 32 x 32
                nn.ReflectionPad2d(1),
                nn.Conv2d(ndf, ndf*2, 4, 2),
                nn.BatchNorm2d(ndf*2),
                nn.LeakyReLU(0.2, inplace=True),
                # the output size is 256 x 16 x 16
                nn.ReflectionPad2d(1),
                nn.Conv2d(ndf*2, ndf*4, 4, 2),
                nn.BatchNorm2d(ndf*4),
                nn.LeakyReLU(0.2, inplace=True),
                # the output size is 512 x 8 x 8
                nn.ReflectionPad2d(1),
                nn.Conv2d(ndf*4, ndf*8, 4, 2),
                nn.BatchNorm2d(ndf*8),
                nn.LeakyReLU(0.2, inplace=True),
                # the output size is 1024 x 4 x 4
                nn.Conv2d(ndf*8, 1, 4, 1, 0),
                nn.Sigmoid()
                # the output is 1 x 1 x 1, represents a probability
                )
    
    def forward(self, input):
        x = self.netd(input)
        return x.view(-1)
    



