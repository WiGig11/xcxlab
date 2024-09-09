import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader,Subset

import torchvision
from torchvision import transforms
import torchvision.transforms as transforms

import pytorch_lightning as pl


import random
from termcolor import colored

# transforms
# prepare transforms 
transform=transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))])
class SCDataset(torch.utils.data.Dataset):
    '''
    this dataset is made for the sc part
    data: source image
    label : output gt
    '''
    def __init__(self, root,transform,target_transform): 
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.datadir = root
        self.labeldir = root
        self.filenames = os.listdir(self.datadir)
        assert len(self.filenames)==len(os.listdir(self.labeldir)) , "data length must equal label length"
        self.filenames = os.listdir(self.datadir)[1:len(self.filenames)-10]
        #self.filenames = os.listdir(self.datadir)[1:128]
        '''
        if self.stage =='train':
            self.datadir = root
            self.labeldir = root
        elif self.stage=='val':
            self.datadir = root+'\\val_data'
            self.labeldir = root+'\\val_label'
        else:
            self.datadir = root+'\\test_data'
            self.labeldir = root+'\\test_label'
        '''
    def __len__(self): 
        return len(self.filenames)
    
    def __getitem__(self,index): 
        imgname = self.datadir + self.filenames[index]
        #print(imgname)
        try:
            data = cv2.imread(imgname)[:, :, ::-1].astype(np.uint8)
            label = cv2.imread(imgname)[:, :, ::-1].astype(np.uint8)
            # heatmap
            data = self.transform(data)
            label = self.target_transform(label)
        except:
            print(imgname)
            pass
        return data,label


class SCDataloader(pl.LightningDataModule):
    def __init__(self, batch_size=64,resize_size = [128,128]):
        super().__init__()
        self.batch_size = batch_size
        self.resize_size= resize_size

    def setup(self,root):
        # transform
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resize_size),
            transforms.ToTensor()
        ])
        target_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resize_size),  # Assuming target needs resizing as well
            transforms.ToTensor()
        ])
        self.all_dataset =  SCDataset(root,data_transform,target_transform=target_transform)
        '''
        if 'train' in stage:
            # assign to use in dataloaders
            self.train_dataset =  SCDataset(root,data_transform,target_transform=target_transform)
        else:
            pass
        if 'val' in stage:
            self.val_dataset = SCDataset(root,data_transform,target_transform=target_transform)
        else:
            pass

        if 'test' in stage:
            self.test_dataset = SCDataset(root,data_transform,target_transform=target_transform)
        else:
            pass
            #raise TypeError("please correct dataset type in 'train,val,test'")
        '''
        # 假设 all_dataset 是您的完整数据集
        total_size = len(self.all_dataset)
        train_ratio = 0.7
        val_ratio = 0.2
        # test_ratio = 0.1 由于三者之和必须为1，测试集比例可以通过1减去其他两者得到
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        #test_size = total_size - train_size - val_size

        # 生成数据集的索引列表
        indices = list(range(total_size))
        # 使用random.seed确保可重复性
        random.seed(0)
        # 打乱索引
        random.shuffle(indices)

        # 根据索引分割数据集
        train_indices = indices[:train_size]
        val_indices = indices[train_size:(train_size + val_size)]
        test_indices = indices[(train_size + val_size):]

        # 使用Subset来创建新的数据集
        
        self.train_dataset = Subset(self.all_dataset, train_indices)
        self.val_dataset = Subset(self.all_dataset, val_indices)
        self.test_dataset = Subset(self.all_dataset, test_indices)
        

    def train_dataloader(self):
        print(colored("Create Train loader Instance",'green'))
        #print("Create Train loader Instance")
        return DataLoader(self.train_dataset, batch_size=self.batch_size,num_workers = 31,shuffle = True)

    def val_dataloader(self):
        print(colored("Create Val loader Instance",'green'))
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers = 31)

    def test_dataloader(self):
        print(colored("Create Test loader Instance",'green'))
        return DataLoader(self.test_dataset, batch_size=self.batch_size,num_workers = 31)

