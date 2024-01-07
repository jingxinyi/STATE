import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedBlock(nn.Module):
    def __init__(self, inchanel, outchannel, stride=1):
        super(GatedBlock, self).__init__()
        self.conv = nn.Conv2d(inchanel, outchannel, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x, y = x.chunk(2, 1)
        x = F.elu(x)
        y = F.sigmoid(y)
        return x * y


class GatedModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GatedModule, self).__init__()
        # self.channel = [in_channel * 2, 16, 32, 32, 32, 16, 12]
        self.channel = [in_channel * 2, 64, 128, 128, 128, 64, 32]
        self.downsample_index = 1
        self.upsample_index = 4

        self.gated_conv_list = torch.nn.ModuleList()
        for i in range(len(self.channel) - 1):
            # if i == self.downsample_index:
            #     self.gated_conv_list.append(GatedBlock(self.channel[i] // 2, self.channel[i + 1], stride=2))
            # elif i == self.upsample_index:
            #     self.gated_conv_list.append(nn.Upsample(scale_factor=2))
            #     self.gated_conv_list.append(GatedBlock(self.channel[i] // 2, self.channel[i + 1]))
            # else:
            #     self.gated_conv_list.append(GatedBlock(self.channel[i] // 2, self.channel[i + 1]))
            self.gated_conv_list.append(GatedBlock(self.channel[i] // 2, self.channel[i + 1]))
        self.conv = nn.Conv2d(self.channel[-1] // 2, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        for module in self.gated_conv_list:
            x = module(x)
        return F.tanh(self.conv(x))
