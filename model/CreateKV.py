import torch
import torch.nn as nn
import numpy as np

class KVcreater(nn.Module):
    def __init__(self, inc = 240, ratio = 4):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.channelwise_pool_A = nn.AdaptiveAvgPool2d((1, 1))
        self.channelwise_pool_M = nn.AdaptiveMaxPool2d((1, 1))

        self.gamma_A = nn.Parameter(torch.zeros(1, inc, 1, 1))
        self.gamma_M = nn.Parameter(torch.zeros(1, inc, 1, 1))

        hidden_dim = inc // ratio

        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=inc, kernel_size=1)
        )

        self.conv_w = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=inc, kernel_size=1)
        )

        self.conv__channelwise_A = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1),
            nn.BatchNorm2d(inc),
            nn.ReLU()
        )
        self.conv__channelwise_M = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1),
            nn.BatchNorm2d(inc),
            nn.ReLU()
        )



    def forward(self, x):
        B, n_patch, hidden = x.size()   # (B, 9, 240)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)  # (B, 240, 3, 3)
        n,c,h,w = x.size()
        x_h = self.pool_h(x)  # (B, 240, 1, 3)
        x_h = self.conv_h(x_h)
        x_w = self.pool_w(x)  # (B, 240, 3, 1)
        x_w = self.pool_w(x_w)
        x_channelwise_A = self.channelwise_pool_A(x)  # (B, 240 ,1 ,1)
        x_channelwise_A = x_channelwise_A * self.gamma_A
        x_channelwise_M = self.channelwise_pool_M(x)   # (B, 240 ,1 ,1)
        x_channelwise_M = x_channelwise_M * self.gamma_M
        x_channel_map = x_channelwise_M + x_channelwise_A
        x_channel_adjust = x * x_channel_map
        x_all_adjust = x_channel_adjust * x_h * x_w
        x_out = x_all_adjust.flatten(2)
        x_out = x_out.permute(0, 2, 1)

        return x_out



# model = KVcreater()
# x = torch.rand(1, 9, 240)
# x = model(x)
# print(x.shape)