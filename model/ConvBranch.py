import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, channel_num=[16, 32, 64, 128]):
        super().__init__()
        self.convblock0 = nn.Sequential(
            nn.Conv2d(in_channels=channel_num[0], out_channels=channel_num[0], kernel_size=7, padding=3),
            nn.BatchNorm2d(channel_num[0]),
            nn.Conv2d(in_channels=channel_num[0], out_channels=channel_num[1], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel_num[1], out_channels=channel_num[0], kernel_size=1)
                    )

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=channel_num[1], out_channels=channel_num[1], kernel_size=7, padding=3),
            nn.BatchNorm2d(channel_num[1]),
            nn.Conv2d(in_channels=channel_num[1], out_channels=channel_num[2], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel_num[2], out_channels=channel_num[1], kernel_size=1)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=channel_num[2], out_channels=channel_num[2], kernel_size=7, padding=3),
            nn.BatchNorm2d(channel_num[2]),
            nn.Conv2d(in_channels=channel_num[2], out_channels=channel_num[3], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel_num[3], out_channels=channel_num[2], kernel_size=1)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=channel_num[3], out_channels=channel_num[3], kernel_size=7, padding=3),
            nn.BatchNorm2d(channel_num[3]),
            nn.Conv2d(in_channels=channel_num[3], out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=channel_num[3], kernel_size=1)
        )


    def forward(self, x0, x1, x2, x3):
        x0_0 = x0 + self.convblock0(x0)
        x0_1 = x0_0 + self.convblock0(x0_0)

        x1_0 = x1 + self.convblock1(x1)
        x1_1 = x1_0 + self.convblock1(x1_0)

        x2_0 = x2 + self.convblock2(x2)
        x2_1 = x2_0 + self.convblock2(x2_0)

        x3_0 = x3 + self.convblock3(x3)
        x3_1 = x3_0 + self.convblock3(x3_0)

        return x0_1, x1_1, x2_1, x3_1




