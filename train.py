# -*- coding: utf-8 -*-
"""
Created on Wed May 27 22:11:41 2020

@author: hunan
"""

import reset_net
import torch
import torch.nn as nn
from dataload import dataload
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
#from multiprocessing import Pool
import torch.multiprocessing as mp
i = 0 
def train_epoch(epoch,net,trainloader,optimizer,loss_function, device):
    global i
    for batch_index, (images, labels) in enumerate(trainloader):
        images = torch.tensor(images, dtype=torch.float32)
        images = Variable(images)
        labels = Variable(labels)

        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch_index % 100 ==0:
            print(loss.cpu().detach().numpy())
            torch.save(net, './Policy/policy{}.pt'.format(i))
            i = i+1

def train(net, path, transform, device, dataloader_kwargs ):

    dataset = dataload(path, transform)
    
    trainloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, **dataloader_kwargs )
    #net= net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001,  weight_decay=5e-4)
    #p_pool = Pool(5)
    for epoch in range(100):
        #p_pool.apply_async(func=train, args=(epoch,))
        train_epoch(epoch,net,trainloader,optimizer,loss_function, device)   
# 网络初始化    

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)
        #nn.init.xavier_uniform(m.bias.data)    

if __name__ == '__main__':    
    path = './Data/train_X'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
    net = reset_net.resnet18()
    net.apply(weights_init)
    net.to(device)
    
    train(net, path, transform, device, dataloader_kwargs )
    
    

'''多进程
    num_processes = 4
    path = './Data/train_X'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
    mp.set_start_method('spawn')
    #trainloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    net = reset_net.resnet10().to(device)
    # NOTE: this is required for the ``fork`` method to work
    net.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(net, path, transform, device, dataloader_kwargs ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
'''     
  