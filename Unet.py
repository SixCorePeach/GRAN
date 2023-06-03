import torch
import torch.nn.functional as F
import torch.nn as nn


class DoubleConvBlock(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(DoubleConvBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        conv_out_1 = self.conv_1(x)
        bn_out_1 = self.bn_1(conv_out_1)
        relu_out = F.relu(bn_out_1)
        conv_out_2 = self.conv_2(relu_out)
        bn_out_2 = self.bn_2(conv_out_2)
        return F.relu(bn_out_2)


class UpSamplingBlock(nn.Module):
    """
            contracting_block_1 = ContractingBlock(3, 64)
            contracting_block_2 = ContractingBlock(3, 64, 3, 1, 1, True)
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True):
        super(UpSamplingBlock, self).__init__()
        self.tran_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x, concat_feature):
        tran_conv_out = self.tran_conv(x)
        return torch.cat((concat_feature, tran_conv_out), dim=1)


class UNet(nn.Module):
    '''UNet  Unet 
    16*20 x 16*16   320 x 256
    lr 0.001
    '''

    def __init__(self, args, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super().__init__()

        self.conv_block_1 = DoubleConvBlock(3, 64)
        self.down_sampling_1 = nn.MaxPool2d(2, 2)
        self.conv_block_2 = DoubleConvBlock(64, 128)
        self.down_sampling_2 = nn.MaxPool2d(2, 2)
        self.conv_block_3 = DoubleConvBlock(128, 256)
        self.down_sampling_3 = nn.MaxPool2d(2, 2)
        self.conv_block_4 = DoubleConvBlock(256, 512)
        self.down_sampling_4 = nn.MaxPool2d(2, 2)
        self.conv_block_5 = DoubleConvBlock(512, 1024)
        self.up_sampling_1 = UpSamplingBlock(1024, 512)
        self.conv_block_6 = DoubleConvBlock(1024, 512)
        self.up_sampling_2 = UpSamplingBlock(512, 256)
        self.conv_block_7 = DoubleConvBlock(512, 256)
        self.up_sampling_3 = UpSamplingBlock(256, 128)
        self.conv_block_8 = DoubleConvBlock(256, 128)
        self.up_sampling_4 = UpSamplingBlock(128, 64)
        self.conv_block_9 = DoubleConvBlock(128, 64)
        self.out = nn.Sequential(nn.Conv2d(64, 2, kernel_size=1), nn.Softmax2d())

    def forward(self, x):
        conv_block_out_1 = self.conv_block_1(x)
        conv_block_out_2 = self.conv_block_2(self.down_sampling_1(conv_block_out_1))
        conv_block_out_3 = self.conv_block_3(self.down_sampling_2(conv_block_out_2))
        conv_block_out_4 = self.conv_block_4(self.down_sampling_3(conv_block_out_3))
        conv_block_out_5 = self.conv_block_5(self.down_sampling_4(conv_block_out_4))

        conv_block_out_6 = self.conv_block_6(self.up_sampling_1(conv_block_out_5, conv_block_out_4))
        conv_block_out_7 = self.conv_block_7(self.up_sampling_2(conv_block_out_6, conv_block_out_3))
        conv_block_out_8 = self.conv_block_8(self.up_sampling_3(conv_block_out_7, conv_block_out_2))
        conv_block_out_9 = self.conv_block_9(self.up_sampling_4(conv_block_out_8, conv_block_out_1))

        out = self.out(conv_block_out_9)

        return out
