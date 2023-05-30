# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import math

import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels, in_channels),
                                      DenseLayer(in_channels*2, in_channels*2),])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels*4, in_channels*4, kernel_size=1)
        
    def forward(self, x):
        return self.lff(self.layers(x))  # local residual learning
        
class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class Attention(nn.Module):
    def __init__(self, reduction=16, n_feats=60):
        super(Attention,self).__init__()
        channel = n_feats
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, input):
        x0 = input
        x1 = self.conv_du(x0)
        output = x1*x0
        return output

class Denseblock(nn.Module):
    def __init__(
        self, conv, input_channel, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        
        super(Denseblock, self).__init__()
        self.dn_layer1 = conv(input_channel, n_feats, kernel_size, bias=bias)
        self.dn_layer2 = conv(n_feats*2, n_feats*3, kernel_size, bias=bias)
        self.act = act
        
    def forward(self, input):
        x1 = self.dn_layer1(input)
        x2 = self.act(x1)
        x2 = torch.cat((x1, x2), dim=1)
        x3 = self.act(self.dn_layer2(x2))
        output = torch.cat((x1, x3), dim=1)
        
        return output
        
class HFBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(HFBlock, self).__init__()
        m = []
        n = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.avgpool = nn.AvgPool2d(2)
        n.append(conv(n_feats, 4 * n_feats, 3, bias))
        n.append(nn.PixelShuffle(2))

        self.body = nn.Sequential(*m)
        self.upsample = nn.Sequential(*n)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res_h = self.avgpool(res)
        res_h = self.upsample(res_h)
        output = res - res_h
        #output = x+output
        
        return output

class HFBlock_1(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(HFBlock_1, self).__init__()
        m = []
        n = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.avgpool = nn.AvgPool2d(2)
        n.append(conv(n_feats, 4 * n_feats, 3, bias))
        n.append(nn.PixelShuffle(2))

        self.body = nn.Sequential(*m)
        self.upsample = nn.Sequential(*n)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res_h = self.avgpool(res)
        res_h = self.upsample(res_h)
        output = res - res_h
        output = x+output
        
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
    
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, input):
        x = input
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return input*self.sigmoid(x)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SPALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SPALayer, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        y = self.sigmoid(x)
        return y * res


class FRDAB(nn.Module):
    def __init__(self, n_feats=32):
        super(FRDAB, self).__init__()

        self.c1 = default_conv(n_feats, n_feats, 1)
        self.c2 = default_conv(n_feats, n_feats // 2, 3)
        self.c3 = default_conv(n_feats, n_feats // 2, 3)
        self.c4 = default_conv(n_feats*2, n_feats, 3)
        self.c5 = default_conv(n_feats // 2, n_feats // 2, 3)
        self.c6 = default_conv(n_feats*2, n_feats, 1)

        self.se = CALayer(channel=2*n_feats, reduction=16)
        self.sa = SPALayer()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x

        y1 = self.act(self.c1(x))

        y2 = self.act(self.c2(y1))
        y3 = self.act(self.c3(y1))

        cat1 = torch.cat([y1, y2, y3], 1)

        y4 = self.act(self.c4(cat1))

        y5 = self.c5(y3)  # 16

        cat2 = torch.cat([y2, y5, y4], 1)

        ca_out = self.se(cat2)
        sa_out = self.sa(cat2)

        y6 = ca_out + sa_out

        y7 = self.c6(y6)

        output = res + y7

        return output       

class FRDB_s(nn.Module):
    def __init__(self, n_feats=32):
        super(FRDB_s, self).__init__()
        
        self.n_feats = n_feats
        self.c1 = default_conv(n_feats, n_feats, 1)
        self.c2 = default_conv(n_feats//2, n_feats // 2, 5)
        self.c3 = default_conv(n_feats//2, n_feats // 2, 5)
        self.c4 = default_conv(n_feats, n_feats, 3)
        self.c5 = default_conv(n_feats // 2, n_feats // 2, 3)
        self.c6 = default_conv(n_feats*2, n_feats, 1)

        self.se = CALayer(channel=2*n_feats, reduction=16)
        self.sa = SPALayer()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x

        y1 = self.act(self.c1(x))
        y2 = torch.split(y1, self.n_feats//2, dim=1)[0]
        y3 = torch.split(y1, self.n_feats//2, dim=1)[1]
        
        y2 = self.act(self.c2(y2))
        y3 = self.act(self.c3(y3))
        
        cat1 = torch.cat([y2, y3], dim=1) + y1
        #cat1 = torch.cat([y1, y2, y3], 1)

        y4 = self.act(self.c4(cat1))

        y5 = self.c5(y3)  # 16

        cat2 = torch.cat([y2, y5, y4], 1)

        #ca_out = self.se(cat2)
        #sa_out = self.sa(cat2)

        y6 = cat2

        y7 = self.c6(y6)

        output = res + y7

        return output  
        
class FRDB(nn.Module):
    def __init__(self, n_feats=32):
        super(FRDB, self).__init__()

        self.c1 = default_conv(n_feats, n_feats, 1)
        self.c2 = default_conv(n_feats, n_feats // 2, 3)
        self.c3 = default_conv(n_feats, n_feats // 2, 3)
        self.c4 = default_conv(n_feats*2, n_feats, 3)
        self.c5 = default_conv(n_feats // 2, n_feats // 2, 3)
        self.c6 = default_conv(n_feats*2, n_feats, 1)

        #self.se = CALayer(channel=2*n_feats, reduction=16)
        #self.sa = SPALayer()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x

        y1 = self.act(self.c1(x))

        y2 = self.act(self.c2(y1))
        y3 = self.act(self.c3(y1))

        cat1 = torch.cat([y1, y2, y3], 1)

        y4 = self.act(self.c4(cat1))

        y5 = self.c5(y3)  # 16

        cat2 = torch.cat([y2, y5, y4], 1)

        #ca_out = self.se(cat2)
        #sa_out = self.sa(cat2)

        y6 = cat2

        y7 = self.c6(y6)

        output = res + y7

        return output   
                
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class Upsampler_x4(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 16 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(4))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler_x4, self).__init__(*m)
        
class Upsampler_int(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Upsample(scale_factor=2, mode='bilinear'))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler_int, self).__init__(*m)