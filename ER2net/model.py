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
'''
modules:
pad fucntion, gated_conv,dilation layer,dilation block,up/down block
'''
def pad(x, ref=None, h=None, w=None):
    assert not (ref is None and h is None and w is None)
    _, _, h1, w1 = x.shape
    if not ref is None:
        _, _, h2, w2 = ref.shape
    else:
        h2, w2 = h, w
    if not h1 == h2 or not w1 == w2:
        x = F.pad(x, (0, w2 - w1, 0, h2 - h1), mode='replicate')
    return x

def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    assert len(image_numpy.shape) in [2, 3] or image_numpy.size == 0
    if len(image_numpy.shape) == 3:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
        image_numpy = image_numpy.astype(imtype)
    else:
        image_numpy = image_numpy * 255.0
        image_numpy = image_numpy.astype(imtype)
    return image_numpy

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
            self.mask  = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(in_channels, out_channels, kernel_size,stride,padding = 0,dilation = dilation,bias=bias),
                nn.Sigmoid()
            )
        else:
            self.mask  = nn.Sequential(
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

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernels=None):
        super(DownBlock, self).__init__()
        
        if isinstance(kernels, int):
            assert mid_channels is None
            self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernels, stride = 2,padding=kernels//2),
            nn.ReLU(inplace=False)
        )
        else:
            if mid_channels is None:
                mid_channels = out_channels
            i_channels = [in_channels] + [mid_channels] * (len(kernels) - 1)
            o_channels = [mid_channels] * (len(kernels) - 1) + [out_channels]
            conv = [nn.Sequential(
                nn.Conv2d(i_channels[0], o_channels[0], kernel_size=kernels[0], stride = 2,padding=kernels[0]//2),
                nn.ReLU(inplace=False)
            )]
            for i in range(1, len(kernels)):
                conv.append(nn.Sequential(
                    nn.Conv2d(i_channels[i], o_channels[i], kernel_size=kernels[i], padding=kernels[i]//2),
                    nn.ReLU(inplace=False)
                ))
            self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, add_channels=None, kernels=None,
                bilinear=False, shape=None):
        super(UpBlock, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        if isinstance(mid_channels, int):
            i_channels = [in_channels] + [mid_channels] * (len(kernels) - 1)
            o_channels = [mid_channels] * (len(kernels) - 1) + [out_channels]
        else:
            assert isinstance(mid_channels, list) or isinstance(mid_channels, tuple)
            assert len(mid_channels) == len(kernels) - 1
            i_channels = [in_channels] + list(mid_channels)
            o_channels = list(mid_channels) + [out_channels]
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if not add_channels is None:
                i_channels[0] = i_channels[0] + add_channels
            conv = []
            for i in range(len(kernels)):
                conv.append(nn.Sequential(
                    nn.Conv2d(i_channels[i], o_channels[i], kernel_size=kernels[i],padding=kernels[i]//2),
                    nn.ReLU(inplace=False)
                ))
            self.conv = nn.Sequential(*conv)
        else:
            self.up = nn.ConvTranspose2d(i_channels[0], o_channels[0], kernel_size=kernels[0], stride=2,
                                         padding=kernels[0] // 2, output_padding=1)
            if not add_channels is None:
                i_channels[1] = i_channels[1] + add_channels
            conv = []
            for i in range(1, len(kernels)):
                conv.append(nn.Sequential(
                    nn.Conv2d(i_channels[i], o_channels[i], kernel_size=kernels[i],padding=kernels[i]//2),
                    nn.ReLU(inplace=False)
                ))
            self.conv = nn.Sequential(*conv)

    def forward(self, x, feat=None, shape=None):
        assert not feat is None or not shape is None
        up = self.up(x)
        up = pad(up, ref=feat, h=None if shape is None else shape[0], w=None if shape is None else shape[1])
        if not feat is None:
            return self.conv(torch.cat([up, feat], dim=1))
        else:
            return self.conv(up)
        
class DilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, dilations):
        super(DilationBlock, self).__init__()
        i_channels = [in_channels] + [mid_channels] * (len(dilations) - 1)
        o_channels = [mid_channels] * len(dilations)
        conv_layers = []
        for i in range(len(dilations)):
            conv_layer = nn.Sequential(
                nn.Conv2d(i_channels[i], o_channels[i], kernel_size=3,padding=dilations[i],dilation=dilations[i]),
                nn.ReLU(inplace=False)
            )
            conv_layers.append(conv_layer)
        self.out = nn.Sequential(
                nn.Conv2d(in_channels + mid_channels, out_channels, kernel_size=1,padding=0,dilation=1),
                nn.ReLU(inplace=False)
            )

    def forward(self, x):
        conv = self.conv_layers(x)
        out = self.out(torch.cat([x, conv], dim=1))
        return out

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

class Encoder_DE(nn.Module):
    def __init__(self):
        super(Encoder_DE,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=False),
            DownBlock(64, 128, kernels=[3, 3]),
            DilationBlock(128, 128, 128, dilations=[2, 4]),
            DownBlock(128, 256, kernels=[3, 3]),
            DilationBlock(256, 256, 256, dilations=[2, 4]),
            res_block(256, blocks=3, resscale=0.1, kernel_size=3, gatedconv=True, dilations=[2, 4])
        )
    def forward(self,image):
        feature = self.encoder(image)
        return feature

class Decoder_DE(nn.Module):
    def __init__(self):
        super(Decoder_DE,self).__init__()
        self.decoder = nn.Sequential(
            UpBlock(256, 128, 128, add_channels=128, kernels=[5, 5], bilinear=True),
            UpBlock(128, 64, 64, add_channels=64, kernels=[5, 5], bilinear=True)
        )#decoder for Detection Branch

    def forward(self,feature):
        res = self.decoder(feature)
        return res

'''
detect branch and elimination branch
'''
class Detection_Elimination(nn.Module):
    def __init__(self):
        super(Detection_Elimination,self).__init__()
        self.encoder = Encoder_DE()
        self.decoder = Decoder_DE()
        self.conv_D = nn.Sequential(
            nn.Conv2d(64+3, 64, kernel_size=3, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )   
        self.conv_E = nn.Sequential(
            nn.Conv2d(64+4, 64, kernel_size=3, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=False)
        )  
    def forward(self,image):
        feature = self.encoder(image)
        decoder_res = self.decoder(feature)
        detection_res = self.conv_D(decoder_res)
        elimination_res = self.conv_D(decoder_res)
        return detection_res,elimination_res,decoder_res

'''
inpainting branch
'''
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
        super(Decoder2, self).__init__()
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
        E_output, softmax_score_query, softmax_score_memory = self.read(E)
        self.update(E,softmax_score_memory)
        return E_output

class Inpainting(nn.Module):
    def __init__(self,encoder2,decoder2,memory_block):
        super(Inpainting,self).__init__()
        self.encoder = encoder2
        self.decoder = decoder2#decoder for Inpainting Branch
        self.memory_block = memory_block

    def forward(self,image,detection_res):
        feature  = self.encoder(image)
        detection_res = cv2.resize(detection_res)
        new_feature = detection_res*feature 
        update_feature = self.memory_block(new_feature)
        inpaint_output = self.decoder(update_feature)
        return inpaint_output

'''
RFM
'''
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
        return x,weight_map

'''
eye flow net
'''
class eyeFlowNet(nn.Module):
    def __init__(self):
        super(eyeFlowNet,self).__init__()
        self.encoder =  nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.1, inplace=False)
            )
        self.decoder =  nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=False),
            nn.Conv2d(128,64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 2, kernel_size=3, padding=1, stride=2),
            nn.Tanh(inplace=False),
            )

    def forward(self,I_eye,I_eye_prime):
        flow = self.decoder(self.encoder(torch.cat([I_eye,I_eye_prime],dim=1)))
        return flow

'''
MTnet
'''
class MTNet(LightningModule):
    def __init__(self,C,M,RFM,eyeFlowNet,L_rec_module,L_p_module,L_focal_module,L_weight_module,L_flow_module,L_eye_module,lr):
        super().__init__()
        self.de = Detection_Elimination()
        self.inpaint = Inpainting(encoder2=Encoder2(),decoder2=Decoder2(),memory_block=memory_Block(C,M))
        self.RFM = RFM
        self.final_removel = nn.Sequential(
            GatedConv(64 + 7, 64, kernel_size=7, padding=3),
            res_block(64, blocks=3, resscale=1.0, kernel_size=3, gatedconv=True, dilations=[2, 4]),
            nn.Sequential(64, 3, kernel_size=3, padding=1),
            nn.ReLU6(inplace=False)
        )
        self.eyeFlowNet = eyeFlowNet
        self.L_rec_module = L_rec_module
        self.L_p_module = L_p_module
        self.L_focal_module = L_focal_module
        self.L_weight_module = L_weight_module
        self.L_flow_module = L_flow_module
        self.L_eye_module = L_eye_module
        self.lr = lr
        self.automatic_optimization = False
        
    def forward(self,image):
        detection,eliminate,decoder_res = self.de(image)
        inpaint = self.inpaint(image,detection)
        _,weight_map  = self.RFM(eliminate,inpaint,detection)
        res = self.final_removel(torch.cat([image, eliminate, decoder_res, detection], dim=1))
        return detection,res,weight_map
                
    def training_step(self,batch):
        optimizer_eye, optimizer_all = self.optimizers()
        current_epoch = self.current_epoch
        image,gt = batch
        if current_epoch<40:
            face_flip_batch = torch.flip(image, dims=[3])  # dims=[3] 表示水平翻转宽度方向
            flow = self.eyeFlowNet(image,face_flip_batch)
            loss = self.L_eye_module(image,face_flip_batch,flow)
            loss = loss.to(self.device)
            optimizer_eye.zero_grad()
            self.manual_backward(loss)
            optimizer_eye.step()
        else:
            detection,res,weight_map = self.forward(image)
            delta = image - gt
            mask = 0.3 * delta[:, 0, :, :] + 0.59 *  delta[:, 1, :, :] + 0.11 * delta[:, 2, :, :]
            mask_max = mask.max(dim=[1, 2], keepdim=True).values  
            mask_binary = (mask > 0.707 * mask_max).float() 
            flow = self.eyeFlowNet(image,face_flip_batch)
            loss = 0
            loss_rec = self.L_rec_module(image,gt)
            loss_p = self.L_p_module(image,gt)*0.01
            loss_focal = self.L_focal_module(detection,mask_binary)
            loss_weight = self.L_weight_module(delta,weight_map)*10
            loss_flow = self.L_flow_module(image,flow)
            face_flip_batch = torch.flip(image, dims=[3])  # dims=[3] 表示水平翻转宽度方向
            loss_eye = self.L_eye_module(image,face_flip_batch,flow)
            loss = loss_rec + loss_p + loss_focal + loss_weight + loss_flow + loss_eye
            loss = loss.to(self.device)
            optimizer_all.zero_grad()
            self.manual_backward(loss)
            optimizer_all.step()

        self.log('training loss',loss,on_step = False, on_epoch = True,prog_bar = True,logger = True, sync_dist=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning rate',current_lr,on_step = False, on_epoch = True,prog_bar = False,logger = True, sync_dist=True)
        return loss          
    
    def validation_step(self,batch,batch_idx):
        image,_ = batch
        _,output_img,_ = self.forward(image)
        self.logger.experiment.add_image('source_img',image[0]/2+0.5,self.current_epoch)
        self.logger.experiment.add_image('outpur_img',output_img[0]/2+0.5,self.current_epoch)

    def configure_optimizers(self):
        optimizer_eye = Adam(self.eyeFlowNet.parameters(), lr=self.lr)
        scheduler_eye = torch.optim.lr_scheduler.StepLR(optimizer_eye,step_size = 5,gamma = 0.4)
        optimizer_all = Adam(self.parameters(), lr=self.lr)
        scheduler_all = torch.optim.lr_scheduler.StepLR(optimizer_all,step_size = 5,gamma = 0.4)
        optim_dict = {'optimizer_eye':optimizer_eye,'lr_scheduler_eye':scheduler_eye,
                    'optimizer_eye':optimizer_all,'lr_scheduler_eye':scheduler_all}
        return optim_dict