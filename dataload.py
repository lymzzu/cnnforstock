# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:52:14 2020

@author: hunan
"""

import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import cv2
class dataload(Dataset):
    def __init__(self,path,transforms):
        super(dataload,self).__init__()
        self.path = path
        self.namelist = os.listdir(self.path)
        self.transforms = transforms
        self.labels = pd.read_csv('./Data/train_label_y.csv', index_col =0)
    def __len__(self):
        return len(self.namelist)
    def __getitem__(self,index):
        self.name = self.namelist[index]
        img = cv2.imread(os.path.join(self.path,self.name))
        imgdata = self.transforms(img)
        label = self.labels['1'][self.name[:-4]]
        return imgdata, label 


if __name__ == '__main__':
    path = './Data/train_X'
    transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), std=(1,1,1))
    ])
    dataset = dataload(path, transforms)
    trainloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    print(dataset[5][0].shape)
    for batch_index, (images, labels) in enumerate(trainloader):
        print (batch_index, (images.shape, labels.shape))
        break
#df = pd.read_csv('./Sample/train_label_y.csv', header = None ,index_col =0)
#img = cv2.imread(os.path.join('./Sample/train_X','sz.300498_30.png'))
