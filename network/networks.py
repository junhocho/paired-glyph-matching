#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import torch.nn as nn

import torch.nn.functional as F

from network.default import DefaultModel, Flatten

__all__ = ['ResNet34', 'ResNet18']

def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None,
               track_running_stats=None, Norm2d=nn.BatchNorm2d):
    super(BasicBlock, self).__init__()

    assert (track_running_stats is not None)

    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = Norm2d(planes, track_running_stats=track_running_stats)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = Norm2d(planes, track_running_stats=track_running_stats)
    self.shortcut = downsample  # XXX changed to support simclr
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.shortcut is not None:
      residual = self.shortcut(x)

    out += residual
    out = self.relu(out)

    return out

class ResNet34(DefaultModel):


    def __init__(self, cin, cout, sobel, net_heads=None, norm_method="BN", use_simclr_head=False):
        # do init
        super(ResNet34, self).__init__()
        # build sobel
        self.sobel = self._make_sobel_() if sobel else None
        # build trunk net
        self.inplanes = 64
        # self.layer1 = nn.Sequential(nn.Conv2d(2 if sobel else cin, 64, 
        #             kernel_size=3, stride=1, padding=1, bias=False),
        #             nn.BatchNorm2d(64, track_running_stats=True),
        #             nn.ReLU(inplace=True),
        #             nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        if norm_method == "BN":
            Norm2d = nn.BatchNorm2d
        elif norm_method == "IN":
            Norm2d = nn.InstanceNorm2d
        else:
            raise RuntimeError("Unknown norm")

        self.conv1 = nn.Conv2d(2 if sobel else cin, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = Norm2d(64, track_running_stats=True)
                # nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        self.layer1 = self._make_layer(BasicBlock, 64, 3, Norm2d=Norm2d)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2, Norm2d=Norm2d)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2, Norm2d=Norm2d)
        self.layer4 = self._make_layer(BasicBlock, cout, 3, stride=2, Norm2d=Norm2d)
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),  Flatten())

        heads = []

        if net_heads:
            if use_simclr_head:
                ## simclr head
                n_feat = cout * BasicBlock.expansion
                heads = [nn.Sequential(
                    nn.Linear(n_feat, n_feat, bias=False),
                    nn.ReLU(),
                    nn.Linear(n_feat, head, bias=False),
                    ) for head in net_heads]
            else:
                ## pui loss head
                heads = [nn.Sequential(
                    nn.Linear(cout * BasicBlock.expansion, head),
                    nn.Softmax(dim=1)
                    ) for head in net_heads]
            self.heads = nn.ModuleList(heads)
        else:
            self.heads = None

    def _make_layer(self, block, planes, blocks, stride=1, Norm2d=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
          downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            Norm2d(planes * block.expansion, 
                        track_running_stats=True))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 
                                    track_running_stats=True, Norm2d=Norm2d))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                    track_running_stats=True, Norm2d=Norm2d))

        return nn.Sequential(*layers)

    # def run(self, x, target=None):
    #     """Function for getting the outputs of intermediate layers
    #     """
    #     # XXX : seems not used function
    #     if target is None or target > 5:
    #         raise NotImplementedError('Target is expected to be smaller than 6')
    #     layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
    #     for layer in layers[:target]:
    #         x = layer(x)
    #     return x

    def inspect_dimension(self, x):
        if self.sobel is not None:
            x = self.sobel(x)

        print("input", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        print("conv1-bn:", x.shape)
        x = self.layer1(x)
        print("layer1:", x.shape)
        x = self.layer2(x)
        print("layer2:", x.shape)
        x = self.layer3(x)
        print("layer3:", x.shape)
        x = self.layer4(x)
        print("layer4:", x.shape)
        x = self.avgpool(x)
        print("avgpool:", x.shape)
        x = list(map(lambda head:head(x), self.heads))
        for i, head in enumerate(self.heads):
            print("head {}:".format(i), x[i].shape)
        return x

    def forward(self, x):
        if self.sobel is not None:
            x = self.sobel(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = self.avgpool(x)
        head_out = list(map(lambda head:head(feat), self.heads)) if self.heads else None

        return feat, head_out

    def image_encode(self, x):
        if self.sobel is not None:
            x = self.sobel(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)   ## /2
        x = self.layer3(x)   ## /4
        x = self.layer4(x)   ## /8

        return x

    def extract_feature(self, x):
        if self.sobel is not None:
            x = self.sobel(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x

    def infer(self, x):
        '''
        Used in test.py
        '''
        if self.sobel is not None:
            x = self.sobel(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        # x = self.avgpool(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)


        # out = map(lambda head:head(x), self.heads)

        outs = []
        for head in self.heads:
            c = head[0].c
            out = head[0](x)        # tp
            out = head[1](out, c)   # mlr
            out = head[2](out)      # softmax
            outs.append(out)
        return outs

        return  x, outs


class ResNet18(DefaultModel):

    def __init__(self, cin, cout, sobel, net_heads=None, norm_method='BN', use_simclr_head=False):
        # net_heads = net_heads if net_heads is not None else cfg.net_heads
        # do init
        super(ResNet18, self).__init__()
        # build sobel
        self.sobel = self._make_sobel_() if sobel else None
        # build trunk net
        self.inplanes = 64

        if norm_method == "BN":
            Norm2d = nn.BatchNorm2d
        elif norm_method == "IN":
            Norm2d = nn.InstanceNorm2d
        else:
            raise RuntimeError("Unknown norm")

        # self.layer1 = nn.Sequential(nn.Conv2d(2 if sobel else cin, 64, 
        #             kernel_size=3, stride=1, padding=1, bias=False),
        #             nn.BatchNorm2d(64, track_running_stats=True),
        #             nn.ReLU(inplace=True),
        #             nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        self.conv1 = nn.Conv2d(2 if sobel else cin, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = Norm2d(64, track_running_stats=True)
                # nn.ReLU(inplace=True),
        self.maxpool =  nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # XXX originally not in SCAN., also maxpool not working in forward
        self.layer1 = self._make_layer(BasicBlock, 64, 2, Norm2d=Norm2d)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, Norm2d=Norm2d)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, Norm2d=Norm2d)
        self.layer4 = self._make_layer(BasicBlock, cout, 2, stride=2, Norm2d=Norm2d)
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),  Flatten())

        heads = []

        if net_heads:
            if use_simclr_head:
                ## simclr head
                n_feat = cout * BasicBlock.expansion
                heads = [nn.Sequential(
                    nn.Linear(n_feat, n_feat, bias=False),
                    nn.ReLU(),
                    nn.Linear(n_feat, head, bias=False),
                    ) for head in net_heads]
            else:
                ## pui loss head
                heads = [nn.Sequential(
                    nn.Linear(cout * BasicBlock.expansion, head),
                    nn.Softmax(dim=1)
                    ) for head in net_heads]
            self.heads = nn.ModuleList(heads)
        else:
            self.heads = None

    def _make_layer(self, block, planes, blocks, stride=1, Norm2d=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
          downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            Norm2d(planes * block.expansion, 
                        track_running_stats=True))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 
                                    track_running_stats=True, Norm2d=Norm2d))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                    track_running_stats=True, Norm2d=Norm2d))

        return nn.Sequential(*layers)

    def inspect_dimension(self, x):
        if self.sobel is not None:
            x = self.sobel(x)

        print("input", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        print("conv1-bn:", x.shape)
        x = self.layer1(x)
        print("layer1:", x.shape)
        x = self.layer2(x)
        print("layer2:", x.shape)
        x = self.layer3(x)
        print("layer3:", x.shape)
        x = self.layer4(x)
        print("layer4:", x.shape)
        x = self.avgpool(x)
        print("avgpool:", x.shape)
        x = list(map(lambda head:head(x), self.heads))
        for i, head in enumerate(self.heads):
            print("head {}:".format(i), x[i].shape)
        return x

    def forward(self, x):
        if self.sobel is not None:
            x = self.sobel(x)

        # torch.Size([256, 3, 32, 32])
        x = F.relu(self.bn1(self.conv1(x)))
        # x = self.maxpool(x)
        # torch.Size([256, 64, 32, 32])
        x = self.layer1(x)
        x = self.layer2(x)
        # torch.Size([256, 128, 16, 16])
        x = self.layer3(x)
        # torch.Size([256, 256, 8, 8])
        x = self.layer4(x)
        # torch.Size([256, 512, 4, 4])
        feat = self.avgpool(x)
        # torch.Size([256, 512, 1, 1])
        
        head_out = list(map(lambda head:head(feat), self.heads)) if self.heads else None

        return feat, head_out


    def image_encode(self, x):
        if self.sobel is not None:
            x = self.sobel(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)   ## /2
        x = self.layer3(x)   ## /4
        x = self.layer4(x)   ## /8

        return x



    def extract_feature(self, x):
        if self.sobel is not None:
            x = self.sobel(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x

    def infer(self, x):
        '''
        Used in test.py
        '''
        if self.sobel is not None:
            x = self.sobel(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        # x = self.avgpool(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)


        # out = map(lambda head:head(x), self.heads)

        outs = []
        for head in self.heads:
            c = head[0].c
            out = head[0](x)        # tp
            out = head[1](out, c)   # mlr
            out = head[2](out)      # softmax
            outs.append(out)
        return outs

        return  x, outs

class MLP(nn.Module):
    def __init__(self, in_channel=512, out_channel=37):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            # nn.Linear(in_channel, in_channel, True),
            # nn.ReLU(inplace=True),
            nn.Linear(in_channel, out_channel, True)
            )

    def forward(self, feature):
        out = self.mlp(feature)
        return out
