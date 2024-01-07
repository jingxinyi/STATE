import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(outchannel)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)))


class DiscriminatorModule(nn.Module):
    def __init__(self):
        super(DiscriminatorModule, self).__init__()
        channel = [3, 32, 64, 128, 256, 256]
        outchannel = 1
        conv_layers = 5
        self.conv_list = nn.ModuleList()
        for i in range(conv_layers):
            if i == conv_layers - 1:
                self.conv = Conv(channel[i], outchannel)
            else:
                self.conv_list.append(Conv(channel[i], channel[i + 1]))

    def forward(self, x):
        for i, module in enumerate(self.conv_list):
            x = F.elu(module(x))
        x = self.conv(x)
        return x