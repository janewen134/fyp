#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
from CSPDarknet import *
import cv2
from config import Cfg

use_attention = Cfg.use_attention

# CBL
def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


#---------------------------------------------------#
#   SPP
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   conv+upsample
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x


#---------------------------------------------------#
#   conv *3
#   [512, 1024]
#---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m



#---------------------------------------------------#
#   conv*5
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   Attention Block
#---------------------------------------------------#    
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        # self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(inp)
        self.bn2 = nn.BatchNorm2d(inp)
        self.ds = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        # self.ds_act1 = nn.ReLU()
        # self.ds_act2 = nn.ReLU()
        
        self.us = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        # self.us_act3 = nn.Sigmoid()
        # self.us_act4 = nn.Sigmoid()

        self.h_swish = h_swish()

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        # print('input size: ', n,c,h,w)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        x_h = self.bn1(x_h)
        x_w = self.bn2(x_w)
        # downsample
        d_h = self.ds(x_h).relu()
        d_w = self.ds(x_w).relu()

        # get attention
        a_h = self.us(d_h).sigmoid()
        a_w = self.us(d_w).sigmoid()

        a_w = a_w.permute(0, 1, 3, 2)

        # res block
        out = identity * a_w * a_h
        out += identity
        out = self.h_swish(out)
        # n1,c1,h1,w1 = out.size()
        # print('output size: ',n1,c1,h1,w1)
        return out

# if __name__ == '__main__':
#     x = torch.randn(2 , 16, 128, 64)    # b, c, h, w
#     ca_model = CA_Block(channel=16, h=128, w=64)
#     y = ca_model(x)
#     print(y.shape)


#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
from torchvision import transforms, datasets as ds
import torchvision as tv
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        #  backbone
        self.backbone = darknet53(None)

        self.conv1 = make_three_conv([512,1024],1024)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512,1024],2048)

        self.upsample1 = Upsample(512,256)
        self.conv_for_P4 = conv2d(512,256,1)
        self.make_five_conv1 = make_five_conv([256, 512],512)

        self.upsample2 = Upsample(256,128)
        self.conv_for_P3 = conv2d(256,128,1)
        self.make_five_conv2 = make_five_conv([128, 256],256)
        # if use_attention:
        #     self.attention2 = CoordAtt(inp=256, oup=256)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 4+1+num_classes
        final_out_filter2 = num_anchors * (5 + num_classes)
        self.yolo_head3 = yolo_head([256, final_out_filter2],128)

        self.down_sample1 = conv2d(128,256,3,stride=2)
        self.make_five_conv3 = make_five_conv([256, 512],512)
        if use_attention:
            self.attention3 = CoordAtt(inp=512, oup=512)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter1 =  num_anchors * (5 + num_classes)
        self.yolo_head2 = yolo_head([512, final_out_filter1],256)


        self.down_sample2 = conv2d(256,512,3,stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024],1024)
        if use_attention:
            self.attention4 = CoordAtt(inp=1024, oup=1024)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter0 =  num_anchors * (5 + num_classes)
        self.yolo_head1 = yolo_head([1024, final_out_filter0],512)



    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)

        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)

        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4,P5_upsample],axis=1)
        P4 = self.make_five_conv1(P4)
        P4_upsample = self.upsample2(P4)
        # print('p4_upsample shape: ', P4_upsample.shape)
        # print('p4_upsample shape 2: ', P4_upsample.shape[2])
        
         # 方法2：plt.imshow(ndarray)
        # img = P4_upsample[0]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
        # img = img.numpy()  # FloatTensor转为ndarray
        # img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
        # # 显示图片
        # plt.imshow(img[:,:,1])
        # plt.show()
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3,P4_upsample],axis=1)
        # if use_attention:
        #     P3 = self.attention2(P3)
        P3 = self.make_five_conv2(P3)

        P3_downsample = self.down_sample1(P3)
        # print('p3_downsample shape: ', P3_downsample.shape)
        P4 = torch.cat([P3_downsample,P4],axis=1)
        if use_attention:
            P4 = self.attention3(P4)
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)
        # print('p4_downsample shape: ', P4_downsample.shape)
        P5 = torch.cat([P4_downsample,P5],axis=1)
        if use_attention:
            P5 = self.attention4(P5)
        P5 = self.make_five_conv4(P5)

        out2 = self.yolo_head3(P3)
        out1 = self.yolo_head2(P4)
        out0 = self.yolo_head1(P5)

        return out0, out1, out2


if __name__ == '__main__':
    # model = YoloBody(3, 80)
    model = YoloBody(3, 20)
    load_all_layers = True
    load_model_pth(model, 'pth/yolo4_weights_my.pth')
    # load_model_pth(model, 'chk/Epoch_007_Loss_34.9990.pth')
    

