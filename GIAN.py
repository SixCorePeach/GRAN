import torch
import math
import torch.nn as nn
__all__ = ['Res2Net', 'res2net50_v1b', 'res2net101_v1b', 'res2net50_v1b_26w_4s']

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
       
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net50_v1b(pretrained=False, **kwargs):
    
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model

def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
    
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model_state = torch.load('/home/sdc_3_7T/jiangyun/CTT/res2net50_v1b_26w_4s-3cf99910.pth')
        model.load_state_dict(model_state)
        # lib.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model

class GCN(nn.Module):
    def __init__(self, num_state, num_htate):
        super(GCN, self).__init__()
        self.num_state = num_state
        self.num_htate = num_htate

        self.relu = nn.ReLU6(inplace=True)
        self.conv = nn.Conv1d(self.num_htate, self.num_htate, kernel_size=1)

    def forward(self, seg, aj):
        n, c, h, w = seg.shape
        seg = seg.view(n*c, h, w).contiguous()
        seg_ = torch.bmm(seg, aj)
        seg_ = seg_.view(n*c, -1, 1).contiguous()
        out = self.relu(self.conv(seg_))
        out = out.view(n*c, h, w).contiguous()
        output = out + seg
        return output


class ETGCN(nn.Module):
    def __init__(self, num_in, num_out, normalize=False):  
        super(ETGCN, self).__init__()
        
        self.num_in = num_in
        self.gcn = GCN(num_state=num_in, num_htate = num_out)
        self.conv_extend = nn.Conv2d(num_in, num_in, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_in)
        self.conv1 = nn.Conv2d(num_in, num_in, kernel_size=1)
        self.conv2 = nn.Conv2d(num_in, num_in, kernel_size=1)
        
    def forward(self, seg, similar):
        
        b, c, h, w = similar.shape
        similar_heng = similar.view(b, c, w, h).contiguous()
        similar_heng = self.conv1(similar_heng)
        similar_heng =  similar_heng.view(b*c, w, h).contiguous()

        similar_shu =  similar.view(b, c, h, w).contiguous()                
        similar_shu = self.conv2(similar_shu)
        similar_shu =  similar_shu.view(b*c, h, w).contiguous()
        similarity = torch.bmm(similar_heng, similar_shu)
        b, c, h, w = seg.shape
        seg_gcn = self.gcn(seg, torch.sigmoid(similarity)).view(b, self.num_in, h, w)
        ext_up_seg_gcn = seg_gcn + seg
        return ext_up_seg_gcn



def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=group, bias=bias)

class SE_Conv_Block(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(SE_Conv_Block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU6(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes * 2)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        self.globalAvgPool = nn.AvgPool2d((10, 8), stride=1)  # (112, 150) for ISIC2018
        self.globalMaxPool = nn.MaxPool2d((10, 8), stride=1)  # (128, 150) for ISIC2017
        self.fc1 = nn.Linear(in_features=inplanes, out_features=round(planes) // 2)
        self.fc2 = nn.Linear(in_features=round(planes) // 2, out_features=inplanes)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2), )

    def forward(self, x):
        residual = x
        # print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)

        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        avg_att = out
        out = out * original_out
        # For global maximum pool
        out1 = self.globalMaxPool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)
        max_att = out1

        out1 = out1 * original_out

        att_weight = avg_att + max_att
        out += out1
        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out, att_weight

class center_extration(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(center_extration, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding = 0)
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding = 1)
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding = 2)
            self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding = 3)
            self.se = SE_Conv_Block(5 * out_channels, 5 * out_channels // 2)
        def forward(self, x):
            out1 = self.conv1(x)
            out2 = self.conv2(x)
            out3 = self.conv3(x)
            out4 = self.conv4(x)
            out0 = out1+out2+out3+out4
            out_end, att = self.se(torch.cat([out0, out1, out2, out3, out4], dim = 1))
            return out_end

class UpSamplingBlock(nn.Module):
    """UNet
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(UpSamplingBlock, self).__init__()
        self.tran_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x, concat_feature):
        tran_conv_out = self.tran_conv(x)
        return torch.cat((concat_feature, tran_conv_out), dim=1)

class DoubleConvBlock(nn.Module):
    """UNet
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DoubleConvBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride = 2, padding = 1)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride = 1, padding = 1)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv_out_1 = self.conv_1(x)
        bn_out_1 = self.bn_1(conv_out_1)
        relu_out = self.relu(bn_out_1)
        conv_out_2 = self.conv_2(relu_out)
        bn_out_2 = self.bn_2(conv_out_2)
        out = self.relu(bn_out_2)
        return out

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride = 1, padding = 1)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride = 1, padding = 1)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv_out_1 = self.conv_1(x)
        bn_out_1 = self.bn_1(conv_out_1)
        relu_out = self.relu(bn_out_1)
        conv_out_2 = self.conv_2(relu_out)
        bn_out_2 = self.bn_2(conv_out_2)
        out = self.relu(bn_out_2)
        return out


class GIAN(nn.Module):
    # double the GCN to check the change of performance
    def __init__(self, args):
        super(GIAN, self).__init__()
        # 256 * 320 -> 64 * 80 -> 16 * 20
        self.bcConv1 = DoubleConvBlock(3, 16)
        self.bcConv2 = DoubleConvBlock(16, 32)
        self.bcConv3 = DoubleConvBlock(32, 32)
        self.bcConv4 = DoubleConvBlock(32, 32)
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.bcConv5 = ConvBlock(3, 256)
        self.bcConv6 = ConvBlock(2048, 512)
        self.bcConv7 = ConvBlock(512, 256)
        
        self.bcConv9 = ConvBlock(256, 128)
        self.bcConv10 = ConvBlock(128, 1)

        self.upConv1 = UpSamplingBlock(320, 128)
        self.upConv2 = UpSamplingBlock(1152, 64)
        self.upConv3 = UpSamplingBlock(576, 64)
        self.upConv4 = UpSamplingBlock(320, 32)
        self.upConv5 = UpSamplingBlock(96, 32)

        self.CE = center_extration(256, 64)
        self.GCN = ETGCN(256, 10*8)
        self.classify = nn.Sequential(nn.Conv2d(35, 2, 1), nn.Softmax2d())
        
    def forward(self, x):
        x1, x2 = x
        out  = self.resnet.conv1(x1.detach().cuda())
        out = self.resnet.bn1(out)
        out = self.resnet.relu(out)
        out_ = out
        out = self.resnet.maxpool(out)      # bs, 64, 160, 128
        
        # ---- low-level features ----
        out1 = self.resnet.layer1(out)      # bs, 256, 80, 64
        out2 = self.resnet.layer2(out1)     # bs, 512, 40, 32
        # ---- high-level features ----
        out3 = self.resnet.layer3(out2)     # bs, 1024, 20, 16
        out4 = self.resnet.layer4(out3)     # bs, 2048, 10, 8
        out4 = self.bcConv7(self.bcConv6(out4))
        Cout_4 = self.bcConv10(self.bcConv9(self.bcConv5(x2.detach().cuda())))
        Cout_4 = Cout_4.expand(-1, out4.shape[1], -1, -1)
        GRAout = self.GCN(out4, Cout_4)
        out = self.CE(GRAout)

        out = self.upConv1(out, out3)
        out = self.upConv2(out, out2)
        out = self.upConv3(out, out1)
        out = self.upConv4(out, out_)
        out = self.upConv5(out, x1.detach().cuda())
        out = self.classify(out)
        return out
