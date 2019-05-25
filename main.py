# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:41:16 2019

@author: hxq
"""
import torch
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader
from model import *
import visdom
import numpy as np
import os
from torchnet.meter import AverageValueMeter

class Config(object):
    data = './data/'     # the road to store data
    ndf = 128                 # the channel of the first convolutional layer of the discriminator net 
    ngf = 128                 # the channel of the last convolutional layer of the generator net
    batch_size = 256
    img_size = 64
    max_epoch = 50            # numbers of iterations
    lr = 1e-4                 # learning rate
    beta = 0.5                # Adam optimizer beat_1
    nz = 100                  # the channel of noise  100 x 1 x 1
    gpu = True                # Use GPU
    d_every = 1               # Train discriminator every 1 batch
    g_every = 5               # Train generator every 5 batch
    vis = True                # Use visdom
    plot_every = 20           # Visdom plot every 20 batch 
    net_path = './checkpoints'
    
    gen_num = 512             # generate 512 images
    gen_select_num = 64       # select best 64 images
    gen_img = 'result.png'
    gen_mean = 0
    gen_std = 1
    
opt = Config()


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    
    device = torch.device('cuda') if opt.gpu else torch.device('cpu')
    
    if opt.vis:
        vis = visdom.Visdom(env="DCGAN")
    
    transforms_ = transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.RandomCrop(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    dataset = tv.datasets.ImageFolder(root=opt.data, transform=transforms_)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    
    # Net
    net_G, net_D = NetG(opt), NetD(opt)
    net_G.to(device)
    net_D.to(device)
    
    # Loss
    criterion = torch.nn.BCELoss().to(device)
    
    # Optimizer
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=opt.lr, betas=(opt.beta, 0.999))
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=opt.lr, betas=(opt.beta, 0.999))
    
    # Set label
    true_labels = torch.ones(opt.batch_size).to(device)
    fake_labels = torch.zeros(opt.batch_size).to(device)
    
    # Set noise images
    noises = torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)  # 256 x 100 x 1 x 1
    
    # Set meter
    D_loss_meter = AverageValueMeter()
    G_loss_meter = AverageValueMeter()
    
    # Train
    for epoch in range(opt.max_epoch):
        for i, (img, _) in enumerate(dataloader):
            real_img = img.to(device)
            bs = len(real_img)
            # Training Discriminator
            if i % opt.d_every == 0:
                optimizer_D.zero_grad()
                
                output_labels = net_D(real_img)
                D_real_loss = criterion(output_labels, true_labels)
                D_real_loss.backward()
                
                fake_img = net_G(noises).detach()   # Truncating the gradient flow of NetG
                fake_output_labels = net_D(fake_img)
                D_fake_loss = criterion(fake_output_labels, fake_labels)
                D_fake_loss.backward()
                
                optimizer_D.step()
                
                D_loss = D_real_loss + D_fake_loss
                D_loss_meter.add(D_loss.item())
            
            # Training Generator
            if i % opt.g_every == 0:
                optimizer_G.zero_grad()
                
                fake_img = net_G(noises)
                fake_output_labels = net_D(fake_img)
                G_loss = criterion(fake_output_labels, true_labels)
                G_loss.backward()
                optimizer_G.step()
                G_loss_meter.add(G_loss.item())
            
            if vis and i % opt.plot_every == 0:
                fix_fake_img = net_G(noises)
                scores = net_D(fix_fake_img)
                show_img = []
                for ii in scores.topk(64)[1]:
                    show_img.append(fix_fake_img.data[ii].detach().cpu() * 0.5 + 0.5)
                vis.images(torch.stack(show_img), win='fake_img')
                vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real_img')
                # vis.line(Y=np.array([D_loss.item()/bs]), X=np.array([i]), win='D_loss', update='append')
                # vis.line(Y=np.array([G_loss.item()/bs]), X=np.array([i]), win='G_loss', update='append')
                
                print('[%d/%d] [%d/%d] D_loss: %.4f G_loss: %.4f ' 
                      % ((epoch+1), opt.max_epoch, i, len(dataloader), D_loss.item()/bs, G_loss.item()/bs))
        
        print('D_loss_meter: %.4f G_loss_meter: %.4f' % (D_loss_meter.value()[0], G_loss_meter.value()[0]))
        vis.line(Y=np.array([D_loss_meter.value()[0]]), X=np.array([epoch]), win='D_loss_meter', update='append')
        vis.line(Y=np.array([G_loss_meter.value()[0]]), X=np.array([epoch]), win='G_loss_meter', update='append')
        D_loss_meter.reset()
        G_loss_meter.reset()
        
        if (epoch+1) == opt.max_epoch:
            if not os.path.exists(opt.net_path):
                os.mkdir(opt.net_path)
            torch.save(net_D.state_dict(), '%s/netd.pth' % opt.net_path)
            torch.save(net_G.state_dict(), '%s/netg.pth' % opt.net_path)
            

def generate(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
        
    device = torch.device('cuda') if opt.gpu else torch.device('cpu')
    
    noises = torch.randn(opt.gen_num, opt.nz, 1, 1).to(device).normal_(opt.gen_mean, opt.gen_std)
    
    net_G, net_D = NetG(opt).eval(), NetD(opt).eval()
    map_location = lambda storage, loc:storage   # load all the tensors into the GPU
    net_D.load_state_dict(torch.load('%s/netd.pth' % opt.net_path, map_location=map_location))
    net_G.load_state_dict(torch.load('%s/netg.pth' % opt.net_path, map_location=map_location))
    net_G.to(device)
    net_D.to(device)
    
    fake_img = net_G(noises)
    scores = net_D(fake_img)
    result = []
    for i in scores.topk(opt.gen_select_num)[1]:
        result.append(fake_img.data[i].detach())
    tv.utils.save_image(torch.stack(result), opt.gen_img, normalize=True, range=(-1,1))
            


if __name__ == '__main__':
    import fire
    fire.Fire()            

    
    
    
    
    



