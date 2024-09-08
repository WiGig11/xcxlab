import torch
import torch.nn as nn
from torch.nn import functional as F

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
#TODO:
'''
#已经完成的部分：Inpaint line
#待完成的部分
1.detect，eliminate头的encoder decoder，这两个是共享encoder，但是有分别的decoder的
2.refined removel
3.loss

'''


#MTNet is a three-branches network, including a Detection Branch, an Inpainting Branch and an Elimination Branch, to predict the reflection detection result D, the content inpainting result Ii and the reflection elimination result Ie, respectively.
class Encoder2(nn.Module):
    def __init__(self, in_channels=3, temp_channels=512):
        super(Encoder2, self).__init__()
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )
        

    def forward(self, x):
        conv1 = self.encoder_conv1(x)
        conv2 = self.encoder_conv2(conv1)
        conv3 = self.encoder_conv3(conv2)
        conv4 = self.encoder_conv4(conv3)
        return conv4
    
class Decoder2(nn.Module):
    def __init__(self, temp_channels=1024, out_channels=3):
        super(Decoder, self).__init__()
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )
        self.up1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2,
                                padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False)
            )
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
        )
        self.up2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2,
                                padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False)
            )
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
        )
        self.up3 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2,
                                padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False)
            )
        self.decoder_conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh(inplace=False)
        )

    def forward(self, fea):
        conv1 = self.decoder_conv1(fea)
        up1 = self.up1(conv1)

        conv2 = self.decoder_conv2(up1)
        up2 = self.up2(conv2)

        conv3 = self.decoder_conv3(up2)
        up3 = self.up3(conv3)

        conv4 = self.decoder_conv4(up3)
        output = self.final_conv(conv4)

        return output
    
class memory_Block(nn.Module):
    def __init__(self,C,M):
        super(memory_Block, self).__init__()
        self.memory = torch.randn(1, 1, C, M)

    def get_score(self, query):
        mem = self.memory
        b, h, w, c = query.size()
        _,_, _, m = mem.size()  
        mem = mem.view(c, m).permute(1, 0)  # (M, C)
        query = query.contiguous().view(b*h*w, c)
        score = torch.matmul(query, torch.t(mem))   # b X h X w X m
        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score,dim=1)
        return score_query, score_memory
    
    def read(self, query):
        batch_size, h, w, dims = query.size()
        softmax_score_query, softmax_score_memory = self.get_score(query)
        E_reshape = query.contiguous().view(batch_size*h*w, dims)
        mem_reshaped = self.memory.view(dims, -1).permute(1, 0)
        E_hat = torch.matmul(softmax_score_memory.detach(), mem_reshaped)# get reweighted memory# score_memory = w_{i,j}
        E_output = torch.cat((E_reshape, E_hat), dim=1)  # (b X h X w) X 2d
        E_output = E_output.view(batch_size, h, w, 2*dims)
        E_output = E_output.permute(0, 3, 1, 2)
        return E_output, softmax_score_query, softmax_score_memory
    
    def update(self,query, softmax_score_memory):
        b,h,w,d = query.shape
        _,_,C,M = self.memory.shape
        reshaped_query = query.contiguous().view(b*h*w, d)
        memory_reshaped = self.memory.view(C, M).permute(1, 0)  # (M, C)
        k_i = torch.argmax(softmax_score_memory, dim=0)  # 输出形状为 (M,)
        mask = torch.zeros(b*h*w, M, dtype=torch.bool)
        mask.scatter_(1, k_i.unsqueeze(1), True)
        weighted_queries = softmax_score_memory.unsqueeze(-1) * reshaped_query.unsqueeze(1)  # (b*h*w, M, C)
        weighted_queries = weighted_queries * mask.unsqueeze(-1)  # (b*h*w, M, C)
        sum_weighted_queries = weighted_queries.sum(dim=0)
        updated_memory = self.memory.view(C, M).permute(1, 0) + sum_weighted_queries  # (M, C)
        updated_memory = updated_memory.permute(1, 0).view(1, 1, C, M)
        updated_memory = F.normalize(updated_memory, p=2, dim=2)
        self.memory = updated_memory
    
    def forward(self, E):
        pass

class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, norm=None,
        num_groups=8, act='elu', negative_slope=0.1, inplace=True, full=True, reflect=True):
        super(GatedConv, self).__init__()
        #conv part 
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size,stride,padding = 0,dilation = dilation,bias=bias),
            nn.ELU(alpha=1.0)
        )
        # mask part
        if full:
            self.mask = self.conv = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(in_channels, out_channels, kernel_size,stride,padding = 0,dilation = dilation,bias=bias),
                nn.Sigmoid()
            )
        else:
            self.mask = self.conv = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(in_channels, 1, kernel_size,stride,padding = 0,dilation = dilation,bias=bias),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.conv(x) * self.mask(x)

class Dilation_Layer(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(Dilation_Layer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear( in_channels * 3,  in_channels * 3 // reduction),
            nn.ELU(inplace=True),
            nn.Linear( in_channels * 3 // reduction,  in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        with torch.no_grad():
            _mean = x.mean(dim=[2, 3])
            _std = x.std(dim=[2, 3])
            _max = x.max(dim=2)[0].max(dim=2)[0]
        feat = torch.cat([_mean, _std, _max], dim=1)
        b, c, _, _ = x.shape
        y = self.fc(feat).view(b, c, 1, 1)
        return x * y
    
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat(
            [F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))
    
class res_block(nn.Module):
    def __init__(self):
        super(res_block, self).__init__()
        self.gated_conv1 = GatedConv(in_channels=256,out_channels=256,kernel_size=2,stride=1,padding = 1)
        self.dilation_layer1 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=2,stride=1,padding = 1)

        self.gated_conv2 = GatedConv(in_channels=256,out_channels=256,kernel_size=2,stride=1,padding = 1)
        self.dilation_layer2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=2,stride=1,padding = 1)

        self.gated_conv3 = GatedConv(in_channels=256,out_channels=256,kernel_size=2,stride=1,padding = 1)
        self.dilation_layer3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=2,stride=1,padding = 1)

        self.ppm = PyramidPooling(256, 256, ct_channels=256)

    def forward(self, x):
        res = self.gated_conv1(x)
        res = self.dilation_layer1(res)
        res1 = res+x*0.1
        res = res+x*0.1

        res = self.gated_conv2(res)
        res = self.dilation_layer2(res)
        res2 = res+res1*0.1
        res = res+res1*0.1

        res = self.gated_conv3(res)
        res = self.dilation_layer3(res)
        res3 = res+res2*0.1

        output = self.ppm(res3)
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
        self.conv1 =  nn.Conv2d(in_channels=7,out_channels=6,kernel_size=1)
        self.conv2 =  nn.Conv2d(in_channels=6,out_channels=6,kernel_size=1)
        self.acti1 = nn.ReLU(inplace=True)
        self.acti2 = nn.Sigmoid()
        #paritial conv part:
        self.conv3 = nn.Conv2d(6, 4, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(6, track_running_stats=False)
        self.acti3 = nn.ReLU(inplace=True)
        self.pooling = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=6,out_channels=3,kernel_size=3,padding = 1,dilation = 1,bias = True)
        self.bn2 = nn.BatchNorm2d(3, track_running_stats=False)

    def forward(self,eliminate_res,inpaint_res,detection_res):
        feature = torch.cat([eliminate_res,inpaint_res,detection_res],dim = 1)
        weight_map = self.conv1(feature)
        weight_map = self.acti(weight_map)
        weight_map = self.conv2(weight_map)
        weight_map = self.acti2(weight_map)
        feature2 = torch.cat([eliminate_res,inpaint_res],dim = 1)# all two for p conv
        x = self.conv3(feature2*weight_map)
        mask_avg = torch.mean(self.pooling(weight_map), dim=1, keepdim=True)
        mask_avg[mask_avg == 0] = 1
        x = x * (1 / mask_avg)
        x = self.bn(x)
        x = self.acti3(x)
        x = self.conv4(x)
        x = self.bn2(x)
        return x
    
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