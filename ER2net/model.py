import torch
import torch.nn as nn

from torch.optim import Adam
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

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

#MTNet is a three-branches network, including a Detection Branch, an Inpainting Branch and an Elimination Branch, to predict the reflection detection result D, the content inpainting result Ii and the reflection elimination result Ie, respectively.

class Decoder(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(Decoder, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding = padding)
        self.activation = nn.PReLU()
        self.gdn = GDN(ch=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.gdn(x)
        x = self.activation(x)
        return x
    
class memory_Block(nn.Module):
    def __init__(self,C,M):
        super(memory_Block, self).__init__()
        self.memory = torch.randn(1, 1, C, M)

    def forward(self, E):
        #x:H W C
        #==read==
        E1 = torch.unsqueeze(E,dim = [2,3])
        weights = softmax(E1,self.memory)
        E_hat = weights.*self.memory#\hat{E}
        output_E = torch.concat(E,E_hat)

        #==update==:
        k_i = argmax(w_{i,j})
        m_j = torch.L2Norm(m_j+sum(indicators(k_i==j)wijei))
        return output_E

class res_block(nn.Module):
    def __init__(self,C,M):
        super(res_block, self).__init__()
        self.gated_conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding = padding)
        self.dilation_layer1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding = padding)
        self.acti1 = xx

        self.gated_conv2 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding = padding)
        self.dilation_layer2 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding = padding)
        self.acti2 = xx

        self.gated_conv3 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding = padding)
        self.dilation_la3 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding = padding)
        self.acti3 = xx

        self.conv =  nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding = padding)

    def forward(self, x):
        res = self.gated_conv1(x)
        res = self.dilation_layer1(res)
        res = self.acti1(res)
        res1 = res+x
        res = res+x

        res = self.gated_conv2(res)
        res = self.dilation_layer2(res)
        res = self.acti2(res)
        res2 = res+res1
        res = res+res1

        res = self.gated_conv3(res)
        res = self.dilation_layer3(res)
        res = self.acti3(res)
        res3 = res+res2

        output = self.conv(res3)
        return output

class Detection(nn.Module):
    def __init__(self,decoder1):
        super(Detection,self).__init__()
        self.decoder = decoder1#decoder for Detection Branch
        
    def forward(self,feature):
        detection_res = self.decoder(feature)
        return detection_res
        
class Inpainting(nn.Module):
    def __init__(self,decoder2,memory_block):
        super(Inpainting,self).__init__()
        self.decoder = decoder2#decoder for Inpainting Branch
        self.memory_block = memory_block

    def forward(self,feature,detection_res):
        detection_res = cv2.resize(detection_res)
        new_feature = detection_res*feature 
        update_feature = self.memory_block(new_feature)
        inpaint_output = self.decoder(update_feature)
        return inpaint_output

class Elimination(nn.Module):
    def __init__(self,decoder3,res_block):
        super(Elimination,self).__init__()
        self.decoder = decoder3#decoder for Elimination Branch
        self.res_block = res_block

    def forward(self,feature):
        decode_res = self.decoder(feature)
        elimination_res = self.res_block(decode_res)
        return elimination_res

class RFM(nn.Module):
    def __init__(self):
        super(RFM,self).__init__()
        self.conv1 =  nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride,padding = padding)
        self.conv2 =  nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride,padding = padding)
        self.conv3 =  nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride,padding = padding)
        self.acti = nn.Sigmoid()
        self.pari_conv = xx

    def forward(self,feature1,feature2,feature3):
        feature = torch.concat(feature1,feature2,feature3)
        weight_map = self.conv1(feature)
        weight_map = self.conv2(weight_map)
        weight_map = self.conv3(weight_map)
        weight_map = self.acti(weight_map)
        I_cat = torch.concat(feature3*weight_map,feature2*(feature1-weight_map))
        I_out = self.pari_conv(I_cat)
        return I_out,weight_map
    
class eyeFlowNet(nn.Module):
    def __init__(self):
        super(eyeFlowNet,self).__init__()
        self.encoder =  nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride,padding = padding)
        self.decoder =  nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride,padding = padding)

    def forward(self,I_eye,I_eye_prime):
        flow = self.decoder(self.encoder(I_eye,I_eye_prime))
        return flow

class MTNet(LightningModule):
    def __init__(self,encoder,detection,inpaint,eliminate,RFM,eye_symmetry_loss_module,fusion_weight_loss_module):
        super().__init__()
        self.encoder = encoder#output H W C
        self.detection = detection#output hwc and resized to HWC
        self.inpaint = inpaint
        self.eliminate = eliminate
        self.RFM = RFM
        self.eye_symmetry_loss_module = eye_symmetry_loss_module
        self.fusion_weight_loss_module = fusion_weight_loss_module
        
    def forward(self,image):
        feature = self.encoder(image)
        detection_res = self.detection(feature)
        inpaint_res = self.inpaint(feature)
        eliminate_res = self.eliminate(feature)
        output_img = self.RFM(detection_res,inpaint_res,eliminate_res)
        return output_img
            
    def training_step(self,batch):
        image,gt = batch
        output_img = self.forward(image)
        flipped_gt = torch.flip(gt)
        eye_symmetry_loss = self.eye_symmetry_loss_module(output_img,flipped_gt)
        fusion_weight_loss = self.fusion_weight_loss_module(output_img,flipped_gt)
        loss = loss.to(self.device)
        self.log('training loss',loss,on_step = False, on_epoch = True,prog_bar = True,logger = True, sync_dist=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning rate',current_lr,on_step = False, on_epoch = True,prog_bar = False,logger = True, sync_dist=True)
        return loss          
    
    def validation_step(self,batch,batch_idx):
        image,gt = batch
        output_img = self.forward(image)
        flipped_gt = torch.flip(gt)
        eye_symmetry_loss = self.eye_symmetry_loss_module(output_img,flipped_gt)
        fusion_weight_loss = self.fusion_weight_loss_module(output_img,flipped_gt)
        loss = loss.to(self.device)
        
        self.logger.experiment.add_image('source_img',image[0]/2+0.5,self.current_epoch)
        self.logger.experiment.add_image('outpur_img',output_img[0]/2+0.5,self.current_epoch)
        self.log('val_loss',loss,on_step = False, on_epoch = True,prog_bar = True,logger = True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        lr_scheduler_type = self.lr_scheduler
        if 'step' in lr_scheduler_type.lower():
            #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones = [500,1000,1500],gamma = 0.1)
            #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones = [500,1000,1500],gamma = 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 400,gamma = 0.1)
        else:
            scheduler = {"scheduler":torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = 10),
                        "monitor":"val loss"}
        optim_dict = {'optimizer':optimizer,'lr_scheduler':scheduler}
        return optim_dict