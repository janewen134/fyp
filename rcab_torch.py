import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

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
        print('input size: ', n,c,h,w)
        x_h = self.pool_h(x)
        # x_w = self.pool_w(x).permute(0, 1, 3, 2)
        x_w = self.pool_w(x)

        x_h = self.bn1(x_h)
        x_w = self.bn2(x_w)
        print('x_h size: ', x_h.size())
        # downsample
        d_h = self.ds(x_h).relu()
        d_w = self.ds(x_w).relu()
        print('d_h size: ', d_h.size())
        # get attention
        a_h = self.us(d_h).sigmoid()
        a_w = self.us(d_w).sigmoid()
        print('a_h size: ', a_h.size())

        # a_w = a_w.permute(0, 1, 3, 2)
        a_w = a_w

        # res block
        out = identity * a_w * a_h
        out += identity
        out = self.h_swish(out)
        n1,c1,h1,w1 = out.size()
        print('output size: ',n1,c1,h1,w1)
        return out

if __name__ == '__main__':
    x = torch.randn(2 , 255, 128, 128)    # b, c, h, w
    ca_model = CoordAtt(inp=255, oup=255)
    y = ca_model(x)
    print(y.shape)