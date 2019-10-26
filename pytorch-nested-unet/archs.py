"""

Refarence: https://github.com/usuyama/pytorch-unet (Unet with Resnet18)
"""

# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out


class UNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(in_channels=args.input_channels,
                                middle_channels=nb_filter[0],
                                out_channels=nb_filter[0]) # [1, 32, 32]
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)


    def forward(self, inputs):
        x0_0 = self.conv0_0(inputs)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(args.input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)


    def forward(self, inputs):
        x0_0 = self.conv0_0(inputs)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.args.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class ResNet18NestedUNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        nb_filter = [64, 64, 128, 256, 512]
        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = nn.Sequential(*self.base_layers[:3])
        self.conv1_0 = nn.Sequential(*self.base_layers[3:5])
        self.conv2_0 = self.base_layers[5]
        self.conv3_0 = self.base_layers[6]
        self.conv4_0 = self.base_layers[7]

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)


    def forward(self, inputs):
        x0_0 = self.conv0_0(inputs)
        x1_0 = self.conv1_0(x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.args.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(self.up(x0_4))
            return output


class ResNet18WithUNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [64, 64, 128, 256, 512]

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.conv0_0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.conv1_0 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)        
        self.conv2_0 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.conv3_0 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.conv4_0 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)


    def forward(self, inputs):
        x0_0 = self.conv0_0(inputs)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        output = self.final(self.up(x0_4))
        return output


class ResNext50NestedUNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        nb_filter = [64, 256, 512, 1024, 2048]
        self.base_model = models.resnext50_32x4d(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = nn.Sequential(*self.base_layers[:3])
        self.conv1_0 = nn.Sequential(*self.base_layers[3:5])
        self.conv2_0 = self.base_layers[5]
        self.conv3_0 = self.base_layers[6]
        self.conv4_0 = self.base_layers[7]

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.args.n_classes, kernel_size=1)


    def forward(self, inputs):
        x0_0 = self.conv0_0(inputs)
        x1_0 = self.conv1_0(x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.args.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(self.up(x0_4))
            return output


class EfficientNetB4NestedUNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        nb_filter = [48, 32, 56, 112, 160, 272, 448]
        self.base_model = EfficientNet.from_pretrained("efficientnet-b4")
        self.base_layers = list(self.base_model.children())

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = nn.Sequential(*self.base_layers[0:1])
        self.conv1_0 = nn.Sequential(*self.base_layers[2][0:5])
        self.conv2_0 = nn.Sequential(*self.base_layers[2][5:10])
        self.conv3_0 = nn.Sequential(*self.base_layers[2][10:15])
        self.conv4_0 = nn.Sequential(*self.base_layers[2][15:20])
        self.conv5_0 = nn.Sequential(*self.base_layers[2][20:25])
        self.conv6_0 = nn.Sequential(*self.base_layers[2][25:])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv4_1 = VGGBlock(nb_filter[4] + nb_filter[5], nb_filter[4], nb_filter[4])
        self.conv5_1 = VGGBlock(nb_filter[5] + nb_filter[6], nb_filter[5], nb_filter[5])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_2 = VGGBlock(nb_filter[3]*2+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv4_2 = VGGBlock(nb_filter[4]*2+nb_filter[5], nb_filter[4], nb_filter[4])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_3 = VGGBlock(nb_filter[2]*3+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_3 = VGGBlock(nb_filter[3]*3+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_4 = VGGBlock(nb_filter[1]*4+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_4 = VGGBlock(nb_filter[2]*4+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_5 = VGGBlock(nb_filter[0]*5+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_5 = VGGBlock(nb_filter[1]*5+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_6 = VGGBlock(nb_filter[0]*6+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], 4, kernel_size=1)


    def forward(self, inputs):
        x0_0 = self.conv0_0(inputs)
        x1_0 = self.conv1_0(x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, x4_0], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        x5_0 = self.conv5_0(x4_0)
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, x4_1], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up(x3_2)], 1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up(x2_3)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up(x1_4)], 1))

        x6_0 = self.conv6_0(x5_0)
        x5_1 = self.conv5_1(torch.cat([x5_0, x6_0], 1))
        x4_2 = self.conv4_2(torch.cat([x4_0, x4_1, self.up(x5_1)], 1))
        x3_3 = self.conv3_3(torch.cat([x3_0, x3_1, x3_2, x4_2], 1))
        x2_4 = self.conv2_4(torch.cat([x2_0, x2_1, x2_2, x2_3, self.up(x3_3)], 1))
        x1_5 = self.conv1_5(torch.cat([x1_0, x1_1, x1_2, x1_3, x1_4, self.up(x2_4)], 1))
        x0_6 = self.conv0_6(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, self.up(x1_5)], 1))

        output = self.final(self.up(x0_6))

        return output


class EfficientNetB4UNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        nb_filter = [48, 32, 56, 112, 160, 272, 448]
        self.base_model = EfficientNet.from_pretrained("efficientnet-b4")
        self.base_layers = list(self.base_model.children())

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if self.args.fpa:
            self.fpa = FPA(channels=56)

        self.conv0_0 = nn.Sequential(*self.base_layers[0:1])  # 42 128 128
        self.conv1_0 = nn.Sequential(*self.base_layers[2][0:5])  # 32 64 64
        self.conv2_0 = nn.Sequential(*self.base_layers[2][5:10])  # 56 32 32
        self.conv3_0 = nn.Sequential(*self.base_layers[2][10:15])  # 112 16 16
        self.conv4_0 = nn.Sequential(*self.base_layers[2][15:20])  # 160 16 16
        self.conv5_0 = nn.Sequential(*self.base_layers[2][20:25])  # 272 8 8
        self.conv6_0 = nn.Sequential(*self.base_layers[2][25:])  # 448 8 8

        self.conv5_1 = VGGBlock(nb_filter[5] + nb_filter[6], nb_filter[5], nb_filter[5])
        self.conv4_2 = VGGBlock(nb_filter[4] + nb_filter[5], nb_filter[4], nb_filter[4])
        self.conv3_3 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        if self.args.fpa:
            self.conv2_4 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        else:
            self.conv2_4 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_5 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_6 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], 4, kernel_size=1)


    def forward(self, inputs):
        x0_0 = self.conv0_0(inputs)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        if self.args.fpa:
            x2_0_fpa = self.fpa(x2_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)
        x5_0 = self.conv5_0(x4_0)
        x6_0 = self.conv6_0(x5_0)

        x5_1 = self.conv5_1(torch.cat([x5_0, x6_0], 1))
        x4_2 = self.conv4_2(torch.cat([x4_0, self.up(x5_1)], 1))
        x3_3 = self.conv3_3(torch.cat([x3_0, x4_2], 1))
        if self.args.fpa:
            x2_4 = self.conv2_4(torch.cat([x2_0_fpa, self.up(x3_3)], 1))
        else:
            x2_4 = self.conv2_4(torch.cat([x2_0, self.up(x3_3)], 1))
        x1_5 = self.conv1_5(torch.cat([x1_0, self.up(x2_4)], 1))
        x0_6 = self.conv0_6(torch.cat([x0_0, self.up(x1_5)], 1))

        output = self.final(self.up(x0_6))

        return output


class EfficientNetB3UNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        nb_filter = [40, 32, 96, 136, 232, 384]
        self.base_model = EfficientNet.from_pretrained("efficientnet-b3")
        self.base_layers = list(self.base_model.children())

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = nn.Sequential(*self.base_layers[0:1])
        self.conv1_0 = nn.Sequential(*self.base_layers[2][0:5])
        self.conv2_0 = nn.Sequential(*self.base_layers[2][5:10])
        self.conv3_0 = nn.Sequential(*self.base_layers[2][10:15])
        self.conv4_0 = nn.Sequential(*self.base_layers[2][15:20])
        self.conv5_0 = nn.Sequential(*self.base_layers[2][20:])

        self.conv4_2 = VGGBlock(nb_filter[4] + nb_filter[5], nb_filter[4], nb_filter[4])
        self.conv3_3 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_4 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_5 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_6 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], 4, kernel_size=1)


    def forward(self, inputs):
        x0_0 = self.conv0_0(inputs)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)
        x5_0 = self.conv5_0(x4_0)

        x4_1 = self.conv4_2(torch.cat([x4_0, x5_0], 1))
        x3_2 = self.conv3_3(torch.cat([x3_0, self.up(x4_1)], 1))
        x2_3 = self.conv2_4(torch.cat([x2_0, x3_2], 1))
        x1_4 = self.conv1_5(torch.cat([x1_0, self.up(self.up(x2_3))], 1))
        x0_5 = self.conv0_6(torch.cat([x0_0, self.up(x1_4)], 1))

        output = self.final(self.up(x0_5))

        return output


class FPA(nn.Module):
    def __init__(self, channels=2048):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(channels/4)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channels)

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Convolution Upsample
        self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)

        # Merge branch 1 and 2
        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))
        x1_merge = self.relu(x1_2 + x2_upsample)

        x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))

        #
        out = self.relu(x_master + x_gpb)

        return out
