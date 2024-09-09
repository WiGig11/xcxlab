import torch
import torch.nn as nn
import torch.nn.functools as F

from torchvision import models

import dlib
import cv2
import numpy as np

import pdb
"------------WeightLoss---------------"
class WeightLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = 0
    def RGB_to_gray(self,RGB):
        return (0.114 * RGB[0, 0, :, :] + 0.587 * RGB[0, 1, :, :] + \
                 0.299 * RGB[0, 2, :, :]).unsqueeze(0).unsqueeze(0)
    
    def forward(self,I_in,I_gt,M):
        R = self.RGB_to_gray(I_in)-self.RGB_to_gray(I_gt)
        t = 0.5*torch.max(R)
        part1 = torch.norm(M[R>t],p=2)**2
        part2 = torch.norm(M[R<t]-1,p=2)**2
        loss = part1+part2
        return loss
    
"------------EyeSymmetryLoss---------------"
class EyeSymmetryLoss(nn.Module):
    def __init__(self):
        super(EyeSymmetryLoss,self).__init__()
        self.detector = dlib.get_frontal_face_detector()

    def forward(self,I_eye,I_eye_prime,flow):
        '''
        I:RGB\BGR
        detector works only on GRAY
        '''
        r = I_eye[:, 2, :, :]  # Red channel
        g = I_eye[:, 1, :, :]  # Green channel
        b = I_eye[:, 0, :, :]  # Blue channel
        face_gray_batch = 0.299 * r + 0.587 * g + 0.114 * b  #TO gray
        
        mask = self.detector(face_gray_batch, 1)
        Omega_x = flow[:, 0, :, :]
        Omega_y = flow[:, 1, :, :]
        batch_size, channels, height, width = I_eye.shape
        grid_x, grid_y = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        grid_x = grid_x.to(I_eye.device).float().unsqueeze(0).expand(batch_size, -1, -1)
        grid_y = grid_y.to(I_eye.device).float().unsqueeze(0).expand(batch_size, -1, -1)
        matched_x = grid_x + Omega_x
        matched_y = grid_y + Omega_y
        matched_x = torch.clamp(matched_x, 0, height - 1)
        matched_y = torch.clamp(matched_y, 0, width - 1)
        grid = torch.stack([matched_y / (width - 1) * 2 - 1, matched_x / (height - 1) * 2 - 1], dim=-1)
        I_gt_sampled = torch.nn.functional.grid_sample(I_eye_prime, grid, align_corners=True)
        l2_loss = torch.mean((I_eye - I_gt_sampled) ** 2, dim=1, keepdim=True)
        loss = l2_loss * mask
        return loss

"------------pixel loss ---------------"
class PxielLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,image,target):
        loss1 = torch.norm(image-target,p=2)**2
        return loss1
    
"----------VGG loss-----------------"
class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i + 1) in indices:
                out.append(X)
        return out


class VGGLoss(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()        
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

"-------focal loss for D------------"
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=[0.25,0.75], gamma=2, ignore_index=None, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

        if self.alpha is None:
            self.alpha = torch.ones(2)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = np.asarray(self.alpha)
            self.alpha = np.reshape(self.alpha, (2))
            assert self.alpha.shape[0] == 2, \
                'the `alpha` shape is not match the number of class'
        elif isinstance(self.alpha, (float, int)):
            self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        else:
            raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target):
        prob = torch.clamp(output, self.smooth, 1.0 - self.smooth)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_loss = -self.alpha[0] * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob) * pos_mask
        neg_loss = -self.alpha[1] * torch.pow(prob, self.gamma) * \
                   torch.log(torch.sub(1.0, prob)) * neg_mask

        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
        num_neg = neg_mask.view(neg_mask.size(0), -1).sum()

        if num_pos == 0:
            loss = neg_loss
        else:
            loss = pos_loss / num_pos + neg_loss / num_neg
        return loss

"---------flow loss -----------"
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

class FlowLoss(nn.Module):
    def __init__(self):
        super(FlowLoss, self).__init__()
        self.loss_F = 0
        self.landmark_loss = 0
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    
    def forward(self,flow,face):#all RGB 
        #BCHW!!
        for i in range(face.shape[0]):
            face1 = tensor2im(face[i])
            face_flip = cv2.flip(face1, 1)
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            flip_gray = cv2.cvtColor(face_flip, cv2.COLOR_BGR2GRAY)
            faces = self.detector(face_gray, 1)
            flip_faces = self.detector(flip_gray, 1)
            for fa, flip_fa in zip(faces, flip_faces):
                keypoint = self.predictor(face, fa)
                flip_keypoint = self.predictor(face_flip, flip_fa)
                for key, flip_key in zip(keypoint.parts(), flip_keypoint.parts()):
                    position = (key.x, key.y)
                    cv2.circle(face, position, 2, (0, 0, 255), -1)
                    flip_position = (flip_key.x, flip_key.y)
                    cv2.circle(face_flip, flip_position, 2, (0, 0, 255), -1)
                # count loss
                for key, flip_key in zip(keypoint.parts(), flip_keypoint.parts()):
                    self.landmark_loss += abs(flow[i][0][flip_key.y - 1][flip_key.x - 1] - key.x) + abs(
                        flow[i][1][flip_key.y - 1][flip_key.x - 1] - key.y)
        self.loss_F += self.landmark_loss * 10.0
        # 求self.flowmap在x和y方向的梯度
        gradxx = flow[..., 1:, :] - flow[..., :-1, :]
        gradxy = flow[..., 1:] - flow[..., :-1]
        gradyx = flow[..., 1:, :] - flow[..., :-1, :]
        gradyy = flow[..., 1:] - flow[..., :-1]
        zero_tensor = torch.zeros(flow[0].shape).to(flow[0].device)
        self.TV_loss = nn.L1Loss(gradxx, zero_tensor) + nn.L1Loss(gradxy, zero_tensor) + nn.L1Loss(gradyx, zero_tensor) + nn.L1Loss(gradyy, zero_tensor)
        self.loss_F += self.TV_loss * 1.0
        return self.loss_F
