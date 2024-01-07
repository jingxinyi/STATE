import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activate_fn=F.elu):
        super(ResidualBlock, self).__init__()
        self.Encoder = Encoder(in_channels, out_channels, activate_fn=activate_fn, is_bn=True)

    def forward(self, x):
        return x + self.Encoder(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activate_fn=F.elu, is_bn=True):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.bn = torch.nn.BatchNorm2d(self.out_channels, affine=True)
        self.activate_fn = activate_fn
        self.is_bn = is_bn

    def forward(self, x):
        x = self.conv(x)

        if self.is_bn:
            x = self.bn(x)

        if self.activate_fn is not None:
            x = self.activate_fn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, activate_fn=F.elu):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        self.Encode = Encoder(in_channels, out_channels, activate_fn=activate_fn)

    def forward(self, x):
        return self.Encode(self.upsample(x))


class FlowModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FlowModule, self).__init__()
        self.residual_out_channel = 512
        self.encode_out_channels = [32, 64, 128, 256, self.residual_out_channel]
        self.decode_out_channels = [256, 128, 64, 32]
        self.residual_num = 4
        self.last_out_channels = [16, 8]

        self.encode_module_list = torch.nn.ModuleList()
        self.residual_module_list = torch.nn.ModuleList()
        self.decode_module_list = torch.nn.ModuleList()

        for i, encode_out_channel in enumerate(self.encode_out_channels):
            encode_in_channel = in_channel if i == 0 else self.encode_out_channels[i - 1]
            self.encode_module_list.append(Encoder(encode_in_channel, encode_out_channel, stride=2))

        for i in range(self.residual_num):
            residual_in_channel = self.encode_out_channels[-1] if i == 0 else self.residual_out_channel
            self.residual_module_list.append(ResidualBlock(residual_in_channel, self.residual_out_channel))

        for i, decode_out_channel in enumerate(self.decode_out_channels):
            decode_in_channel = self.residual_out_channel if i <= 1 else self.decode_out_channels[i - 2]
            self.decode_module_list.append(Decoder(decode_in_channel, decode_out_channel))

        self.last_deconv_1 = Decoder(self.decode_out_channels[-1], self.last_out_channels[0])
        self.last_deconv_2 = Decoder(self.last_out_channels[0], self.last_out_channels[1], scale_factor=1)
        self.last_deconv_final = Decoder(self.last_out_channels[-1], out_channel, scale_factor=1, activate_fn=None)

    def forward(self, x):
        skip = []

        for module in self.encode_module_list:
            x = module(x)
            skip.append(x)
        for module in self.residual_module_list:
            x = module(x)
        for i, module in enumerate(self.decode_module_list):
            x = torch.cat((x, skip[len(self.encode_out_channels) - 1 - i]), 1) if i > 0 else x
            x = module(x)
        x = self.last_deconv_1(x)
        x = self.last_deconv_2(x)

        return self.last_deconv_final(x)
