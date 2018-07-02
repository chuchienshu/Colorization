from __future__ import print_function
import math
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from models.custom_layers.trainable_layers import *

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def load_state_dict(self, state_dict, strict=True):

        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                try:
                    if 'features.0' in name:
                        continue
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 64
    count = 0
    downsample = [1, 3, 6]
    for v in cfg:
        if v == 'M':
            # Maxpool here actually doesnot work,just convenient to load state dict!
            layers += [nn.MaxPool2d(kernel_size=1, stride=1)]
        else:
            
            if count in downsample:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3,stride=2, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            count += 1
            in_channels = v
    return nn.Sequential(*layers)

def conv( in_c, out_c,blocks, strides,  kernel_size=3,batchNorm=True, bias=True):

    model = []
    assert len(strides) == blocks

    for i in range(blocks):
        model += [nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=strides[i], padding=(kernel_size-1)//2, bias=bias),
            nn.ReLU()]
        in_c = out_c

    if batchNorm:
        model += [nn.BatchNorm2d(out_c)]
        return nn.Sequential(*model)

    else:
        return nn.Sequential(*model)

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
#cfg = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
model_path = 'pretrained/vgg19_bn-c79401a0.pth'

class ColorizationNet(nn.Module):
    def __init__(self, batchNorm=True, pretrained=True):
        super().__init__()

        self.nnecnclayer = NNEncLayer()
        self.priorboostlayer = PriorBoostLayer()
        self.nongraymasklayer = NonGrayMaskLayer()
        # self.rebalancelayer = ClassRebalanceMultLayer()
        self.rebalancelayer = Rebalance_Op.apply
        # Rebalance_Op.apply
        self.pool = nn.AvgPool2d(4,4)
        self.upsample = nn.Upsample(scale_factor=4)

        self.bw_conv = nn.Conv2d(1,64,3, padding=1)
        self.main = VGG(make_layers(cfg, batch_norm=batchNorm))

        if pretrained:
            print('loading pretrained model....')
            self.main.load_state_dict(torch.load( model_path))

        self.main.classifier = nn.ConvTranspose2d(512,256,4,2, padding=1)
        self.relu = nn.ReLU()

        self.conv_8 = conv(256,256,2,[1,1], batchNorm=False)
        self.conv313 = nn.Conv2d(256,313,1,1)
        
    def forward(self, gt_img):
        gt_img_l = (gt_img[:,:1,:,:] - 50.) * 0.02
        x = self.bw_conv(gt_img_l)
        x = self.relu (self.main(x))
        x = self.conv_8(x)
        gen = self.conv313(x)

        # ********************** process gtimg_ab *************
        gt_img_ab = self.pool(gt_img[:,1:,:,:]).cpu().data.numpy()
        
        enc = self.nnecnclayer(gt_img_ab)
        
        ngm = self.nongraymasklayer(gt_img_ab)
        pb = self.priorboostlayer(enc)
        boost_factor = (pb * ngm).astype('float32')
        boost_factor = Variable(torch.from_numpy(boost_factor).cuda())

        wei_output = self.rebalancelayer(gen, boost_factor)
        if self.training:
            
            return wei_output, Variable(torch.from_numpy(enc).cuda())
        else:
            return self.upsample(gen),wei_output, Variable(torch.from_numpy(enc).cuda())
