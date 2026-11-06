import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer, nonlinearity='leaky_relu'):
    '''
    Initialize a layer
    '''
    classname = layer.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        # nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
        
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0.0)

# loss_diff
class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()
    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        
        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)
        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        return diff_loss
    
class CMD(nn.Module):
    """
    loss_sim
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """
    def __init__(self, n_moments):
        super(CMD, self).__init__()
        self.n_moments = n_moments
        
    def forward(self, x1, x2):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(self.n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size=(3,3), stride=(1,1), padding=(1,1),
                dilation=1, bias=False):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=out_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.init_weights()
        
    def init_weights(self):
        for layer in self.double_conv:
            init_layer(layer)
        
    def forward(self, x):
        x = self.double_conv(x)

        return x
    
class ConvUpSampler(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size=(3,3), stride=(1,1), padding=(1,1),
                dilation=1, activator=nn.ReLU(), bias=False):
        super().__init__()

        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
        )
        if activator is not None:
            self.up_conv.add_module(name='activation', module=activator)
    
        # self.init_weights()
        
    def init_weights(self):
        for layer in self.up_conv:
            init_layer(layer)
        
    def forward(self, x):
        x = self.up_conv(x)
        return x

class ABSMutualInformation(nn.Module):
    def __init__(self):
        super().__init__()

    def entropy(self, x):
        # 计算熵
        probs = F.softmax(x, dim=-1)
        log_probs = torch.log(probs + 1e-20)  
        return -torch.sum(probs * log_probs, dim=-1)

    def joint_entropy(self, x, y):
        # 计算联合熵
        xy = torch.cat([x, y], dim=-1)
        return self.entropy(xy)

    def forward(self, x, y):
        # 计算互信息
        return torch.abs((self.entropy(x) + self.entropy(y) - self.joint_entropy(x, y)).mean())

class InfoReduce(nn.Module):
    def __init__(self):
        super().__init__()

    def entropy(self, x):
        # 计算熵
        probs = F.softmax(x, dim=-1)
        log_probs = torch.log(probs + 1e-20)  
        return -torch.sum(probs * log_probs, dim=-1)

    def conditional_entropy(self, x, c):
        # 计算条件熵 H(x|y)
        # 假设 y 是 one-hot 编码的
        prob_y = F.softmax(c, dim=-1)
        prob_xy = F.softmax(x + c, dim=-1)
        log_prob_xy = torch.log2(prob_xy + 1e-12)
        conditional_entropy = -torch.sum(prob_y * log_prob_xy, dim=-1)
        return conditional_entropy

    def forward(self, x, c):
        # 计算x关于yx的信息增益：(还是应该关于xy?)
        b, ch, t, d = x.shape
        x_ = x.mean(-1).permute(0, 2, 1).contiguous().view(-1, ch) # b, t, c ; view --> b * t, c
        c = c.squeeze(-1).permute(0, 2, 1).contiguous().view(-1, ch)
        ig = self.entropy(x_) - self.conditional_entropy(x_, c)
        ig = ig.view(b, 1, t, 1)
        return x - ig


class PositionalEncoding(nn.Module):
    def __init__(self, pos_len, d_model=512, pe_type='t', dropout=0.0):
        """ Positional encoding using sin and cos

        Args:
            pos_len: positional length
            d_model: number of feature maps
            pe_type: 't' | 'f' , time domain, frequency domain
            dropout: dropout probability
        """
        super().__init__()
        
        self.pe_type = pe_type
        pe = torch.zeros(pos_len, d_model)
        pos = torch.arange(0, pos_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = 0.1 * torch.sin(pos * div_term)
        pe[:, 1::2] = 0.1 * torch.cos(pos * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2) # (N, C, T)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        # x is (N, C, T, F) or (N, C, T) or (N, C, F)
        if x.ndim == 4:
            if self.pe_type == 't':
                pe = self.pe.unsqueeze(3)
                x += pe[:, :, :x.shape[2]]
            elif self.pe_type == 'f':
                pe = self.pe.unsqueeze(2)
                x += pe[:, :, :, :x.shape[3]]
        elif x.ndim == 3:
            x += self.pe[:, :, :x.shape[2]]
        return self.dropout(x)

