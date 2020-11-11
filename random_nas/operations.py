import torch
import torch.nn as nn

def identity(in_channels, out_channels, stride=1):
    return nn.Identity()

def avgpool3x3(in_channels, out_channels, stride=1):
    return nn.AvgPool2d(kernel_size=(3,3), stride=stride)

def maxpool3x3(in_channels, out_channels, stride=1):
    return nn.MaxPool2d(kernel_size=(3,3), stride=stride)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, groups=1, kernel_size=(1,3), stride=stride,
                     padding=1, bias=False)

def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, groups=1, kernel_size=(1,5), stride=stride,
                     padding=1, bias=False)

def sep3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, groups=in_channels, kernel_size=(3,1), stride=stride,
                     padding=1, bias=False)

def sep5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, groups=in_channels, kernel_size=(5,1), stride=stride,
                     padding=1, bias=False)

def add(a, b):
    print(a.size())
    print(b.size())
    return a + b

def concat(a, b):
    print(a.size())
    print(b.size())
    return torch.cat((a, b), dim=1)
