from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class upConv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(upConv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
        )

    def forward(self,x):
        x = self.up(x)
        return x

class convBlock(nn.Module):
    def __init__(self,ch_in,ch_out , dilation = 1 , padding = 1):
        super(convBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, dilation=dilation , padding=padding, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, dilation=dilation, padding=padding, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, dilation = 2, padding = 2):
        super(UNet, self).__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = convBlock(ch_in=img_ch, ch_out=64, dilation = dilation , padding=padding)
        self.conv2 = convBlock(ch_in=64, ch_out=128, dilation = dilation , padding=padding)
        self.conv3 = convBlock(ch_in=128, ch_out=256, dilation = dilation , padding=padding)
        self.conv4 = convBlock(ch_in=256, ch_out=512, dilation = dilation , padding=padding)
        self.conv5 = convBlock(ch_in=512, ch_out=1024, dilation = dilation , padding=padding)

        self.up5 = upConv(ch_in=1024, ch_out=512)
        self.upConv5 = convBlock(ch_in=1024, ch_out=512)

        self.up4 = upConv(ch_in=512, ch_out=256)
        self.upConv4 = convBlock(ch_in=512, ch_out=256)

        self.up3 = upConv(ch_in=256, ch_out=128)
        self.upConv3 = convBlock(ch_in=256, ch_out=128)

        self.up2 = upConv(ch_in=128, ch_out=64)
        self.upConv2 = convBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoder
        e1 = self.conv1(x)

        e2 = self.maxPool(e1)
        e2 = self.conv2(e2)

        e3 = self.maxPool(e2)
        e3 = self.conv3(e3)

        e4 = self.maxPool(e3)
        e4 = self.conv4(e4)

        e5 = self.maxPool(e4)
        e5 = self.conv5(e5)

        # decoder
        d5 = self.up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.upConv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.upConv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.upConv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.upConv2(d2)
        
        d1 = self.conv_1x1(d2)

        return torch.sigmoid(d1)

class UNetDenoised(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(UNetDenoised,self).__init__()

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = convBlock(ch_in=img_ch, ch_out=64)
        self.conv2 = convBlock(ch_in=64, ch_out=128)
        self.conv3 = convBlock(ch_in=128, ch_out=256)
        self.conv4 = convBlock(ch_in=256, ch_out=512)
        self.conv5 = convBlock(ch_in=512, ch_out=1024)

        self.up5 = upConv(ch_in=1024, ch_out=512)
        self.upConv5 = convBlock(ch_in=1024, ch_out=512)

        self.up4 = upConv(ch_in=512, ch_out=256)
        self.upConv4 = convBlock(ch_in=512, ch_out=256)

        self.up3 = upConv(ch_in=256, ch_out=128)
        self.upConv3 = convBlock(ch_in=256, ch_out=128)

        self.up2 = upConv(ch_in=128, ch_out=64)
        self.upConv2 = convBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.Denoiser = nn.Sequential( OrderedDict([
            ("conv_1" , nn.Conv2d(1 , 1 , 3 , padding=1)),
            ("relu_1" , nn.LeakyReLU()),
            ("conv_2" , nn.Conv2d(1 , 1 , 3 , padding=1)),
            ("relu_2" , nn.LeakyReLU()),
            ("conv_3" , nn.Conv2d(1 , 1 , 3 , padding=1)),
            ("relu_3" , nn.LeakyReLU()),
            ("conv_4" , nn.Conv2d(1 , 1 , 3 , padding=1)),
            ("sigmoid" , nn.Sigmoid())
        ]))

    def forward(self, x):

        # encoder
        e1 = self.conv1(x)

        e2 = self.maxPool(e1)
        e2 = self.conv2(e2)

        e3 = self.maxPool(e2)
        e3 = self.conv3(e3)

        e4 = self.maxPool(e3)
        e4 = self.conv4(e4)

        e5 = self.maxPool(e4)
        e5 = self.conv5(e5)

        # decoder
        d5 = self.up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.upConv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.upConv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.upConv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.upConv2(d2)

        d1 = self.conv_1x1(d2)

        return self.Denoiser(d1)

class UNetBranched(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, dilation = 1, padding = 1):
        super(UNetBranched, self).__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_branch_1 = convBlock(ch_in=img_ch, ch_out=64, dilation = dilation , padding=padding)
        self.conv2_branch_1 = convBlock(ch_in=64, ch_out=128, dilation = dilation , padding=padding)
        self.conv3_branch_1 = convBlock(ch_in=128, ch_out=256, dilation = dilation , padding=padding)
        self.conv4_branch_1 = convBlock(ch_in=256, ch_out=512, dilation = dilation , padding=padding)
        self.conv5_branch_1 = convBlock(ch_in=512, ch_out=1024, dilation = dilation , padding=padding)

        self.conv1_branch_2 = convBlock(ch_in=img_ch, ch_out=64, dilation = dilation , padding=padding)
        self.conv2_branch_2 = convBlock(ch_in=64, ch_out=128, dilation = dilation , padding=padding)
        self.conv3_branch_2 = convBlock(ch_in=128, ch_out=256, dilation = dilation , padding=padding)
        self.conv4_branch_2 = convBlock(ch_in=256, ch_out=512, dilation = dilation , padding=padding)
        self.conv5_branch_2 = convBlock(ch_in=512, ch_out=1024, dilation = dilation , padding=padding)

        self.up5 = upConv(ch_in=1024, ch_out=512)
        self.upConv5 = convBlock(ch_in=1024, ch_out=512)

        self.up4 = upConv(ch_in=512, ch_out=256)
        self.upConv4 = convBlock(ch_in=512, ch_out=256)

        self.up3 = upConv(ch_in=256, ch_out=128)
        self.upConv3 = convBlock(ch_in=256, ch_out=128)

        self.up2 = upConv(ch_in=128, ch_out=64)
        self.upConv2 = convBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x , x_1 = x

        # encoder - branch 2
        e1_branch_2 = self.conv1_branch_2(x_1)

        e2_branch_2 = self.maxPool(e1_branch_2)
        e2_branch_2 = self.conv2_branch_2(e2_branch_2)

        e3_branch_2 = self.maxPool(e2_branch_2)
        e3_branch_2 = self.conv3_branch_2(e3_branch_2)

        e4_branch_2 = self.maxPool(e3_branch_2)
        e4_branch_2 = self.conv4_branch_2(e4_branch_2)

        e5_branch_2 = self.maxPool(e4_branch_2)
        e5_branch_2 = self.conv5_branch_2(e5_branch_2)

        # encoder - branch 1
        e1_branch_1 = self.conv1_branch_1(x)
        e1_branch_1 = e1_branch_1+e1_branch_2

        e2_branch_1 = self.maxPool(e1_branch_1)
        e2_branch_1 = self.conv2_branch_1(e2_branch_1)
        e2_branch_1 = e2_branch_1+e2_branch_2

        e3_branch_1 = self.maxPool(e2_branch_1)
        e3_branch_1 = self.conv3_branch_1(e3_branch_1)
        e3_branch_1 = e3_branch_1+e3_branch_2

        e4_branch_1 = self.maxPool(e3_branch_1)
        e4_branch_1 = self.conv4_branch_1(e4_branch_1)
        e4_branch_1 = e4_branch_1+e4_branch_2

        e5_branch_1 = self.maxPool(e4_branch_1)
        e5_branch_1 = self.conv5_branch_1(e5_branch_1)
        e5_branch_1 = e5_branch_1+e5_branch_2

        # decoder
        d5 = self.up5(e5_branch_1)
        d5 = torch.cat((e4_branch_1, d5), dim=1)
        d5 = self.upConv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((e3_branch_1, d4), dim=1)
        d4 = self.upConv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e2_branch_1, d3), dim=1)
        d3 = self.upConv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1_branch_1, d2), dim=1)
        d2 = self.upConv2(d2)

        d1 = self.conv_1x1(d2)

        return torch.sigmoid(d1)
