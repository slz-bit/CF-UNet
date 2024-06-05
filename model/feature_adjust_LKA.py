import torch.nn as nn
import torch
from timm.models.layers import DropPath


class FeatureAdjust(nn.Module):
    def __init__(self, channel_num=[16, 32, 64, 128]):
        super(FeatureAdjust, self).__init__()

        self.adjustbranch0 = nn.Sequential(
            nn.Conv2d(in_channels=channel_num[0], out_channels=channel_num[0], kernel_size=3, padding=1, stride=2),
            nn.Conv2d(in_channels=channel_num[0], out_channels=channel_num[2], kernel_size=1),
            nn.BatchNorm2d(channel_num[2]),
            nn.ReLU()
        )

        self.adjustbranch1 = nn.Sequential(
            nn.Conv2d(in_channels=channel_num[1], out_channels=channel_num[1], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=channel_num[1], out_channels=channel_num[2], kernel_size=1),
            nn.BatchNorm2d(channel_num[2]),
            nn.ReLU()
        )

        self.adjustbranch2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=channel_num[2], out_channels=channel_num[2], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=channel_num[2], out_channels=channel_num[2], kernel_size=1),
            nn.BatchNorm2d(channel_num[2]),
            nn.ReLU()
        )

        self.adjustbranch3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=channel_num[3], out_channels=channel_num[3], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=channel_num[3], out_channels=channel_num[2], kernel_size=1),
            nn.BatchNorm2d(channel_num[2]),
            nn.ReLU()
        )

        self.conv = nn.Conv2d(channel_num[2]*4, channel_num[2], kernel_size=1)



    def forward(self, x0, x1, x2, x3):
        x0 = self.adjustbranch0(x0)
        x1 = self.adjustbranch1(x1)
        x2 = self.adjustbranch2(x2)
        x3 = self.adjustbranch3(x3)
        feature = self.conv(torch.cat((x0, x1, x2, x3), dim=1))

        return x0, x1, x2, x3, feature



class MSLK(nn.Module):
    def __init__(self, kernel=[9, 11, 13], inc=64):
        super(MSLK, self).__init__()

        self.dwconv0 = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=(1, kernel[0]), padding=(0,kernel[0]//2)),
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=(kernel[0], 1), padding=(kernel[0]//2, 0))
        )

        self.dwconv1 = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=(1, kernel[1]), padding=(0,kernel[1]//2)),
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=(kernel[1], 1), padding=(kernel[1]//2, 0))
        )

        self.dwconv2 = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=(1, kernel[2]), padding=(0,kernel[2]//2)),
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=(kernel[2], 1), padding=(kernel[2]//2, 0))
        )

        self.conv1 = nn.Conv2d(in_channels=inc * 3, out_channels=inc, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(inc)
        self.act = nn.ReLU()

    def forward(self, feature):
        feature_self = feature.clone()
        feature_0 = self.dwconv0(feature)
        feature_1 = self.dwconv1(feature)
        feature_2 = self.dwconv2(feature)
        feature_afterMSLK = self.conv1(torch.cat((feature_0, feature_1, feature_2), dim=1))
        feature_reparameterize = self.conv3(feature_self)
        feature_3 = self.act(self.bn(feature_reparameterize + feature_afterMSLK))
        feature_map = feature_3 + feature_self

        # print(feature_map.shape)

        return feature_map


class FeatureRestore(nn.Module):
    def __init__(self, channel_num=[16, 32, 64, 128]):
        super(FeatureRestore, self).__init__()
        self.restore0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=channel_num[2], out_channels=channel_num[0], kernel_size=1)
        )
        self.restore1 = nn.Conv2d(in_channels=channel_num[2], out_channels=channel_num[1], kernel_size=1)
        self.restore2 = nn.AdaptiveAvgPool2d((64, 64))
        self.restore3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Conv2d(in_channels=channel_num[2], out_channels=channel_num[3], kernel_size=1)
        )

    def forward(self, x0, x1, x2, x3):
        x0 = self.restore0(x0)
        x1 = self.restore1(x1)
        x2 = self.restore2(x2)
        x3 = self.restore3(x3)

        return x0, x1, x2, x3




class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.adjust = FeatureAdjust(channel_num=[16, 32, 64, 128])
        self.attn = MSLK(kernel=[9, 11, 13], inc=64)
        self.restore = FeatureRestore(channel_num=[16, 32, 64, 128])
        layer_scale_init_value = 1e-2
        self.layer_scale_0 = nn.Parameter(
            layer_scale_init_value * torch.ones((64)), requires_grad=True)
        # print(self.layer_scale_0.shape)
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((64)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((64)), requires_grad=True)
        self.layer_scale_3 = nn.Parameter(
            layer_scale_init_value * torch.ones((64)), requires_grad=True)

    def forward(self, x0, x1, x2, x3):
        x0_ad, x1_ad, x2_ad, x3_ad, feature = self.adjust(x0, x1, x2, x3)
        # print(x0_ad.shape)
        feature_map = self.attn(feature)
        x0_attn = self.layer_scale_0.unsqueeze(-1).unsqueeze(-1) * feature_map * x0_ad
        x1_attn = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * feature_map * x1_ad
        x2_attn = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * feature_map * x2_ad
        x3_attn = self.layer_scale_3.unsqueeze(-1).unsqueeze(-1) * feature_map * x3_ad

        x0_restore, x1_restore, x2_restore, x3_restore = self.restore(x0_attn, x1_attn, x2_attn, x3_attn)

        return x0_restore, x1_restore, x2_restore, x3_restore


# model =  FeatureFusion()
# x0 = torch.rand(1, 16, 128, 128)
# x1 = torch.rand(1, 32, 64, 64)
# x2 = torch.rand(1, 64, 32, 32)
# x3 = torch.rand(1, 128, 16, 16)
# x0, x1, x2, x3= model(x0, x1, x2, x3)
# print('x0+' + str(x0.shape))
# print('x1+' + str(x1.shape))
# print('x2+' + str(x2.shape))
# print('x3+' + str(x3.shape))