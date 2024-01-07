import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module_common import bilinear_sampler
from torchvision import transforms


class ConvDown(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ConvDown, self).__init__()
        self.conv1 = nn.Conv2d(inchannel,
                               outchannel,
                               kernel_size=4,
                               stride=2,
                               padding=1)
        self.norm = nn.InstanceNorm2d(outchannel)
        self.conv2 = nn.Conv2d(outchannel,
                               outchannel,
                               kernel_size=3,
                               stride=1,
                               padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.norm(self.conv1(x)))
        x = F.leaky_relu(self.norm(self.conv2(x)))
        return x


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel,
                               outchannel,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(outchannel,
                               outchannel,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.norm = nn.InstanceNorm2d(outchannel)

    def forward(self, x):
        source = x
        x = F.leaky_relu(self.norm(self.conv1(x)))
        x = self.norm(self.conv2(x))
        return F.leaky_relu(x + source)


class ResBlockUp(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResBlockUp, self).__init__()
        self.conv1 = nn.Conv2d(inchannel,
                               inchannel,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.norm1 = nn.InstanceNorm2d(inchannel)
        self.conv2 = nn.ConvTranspose2d(inchannel,
                                        outchannel,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1)
        self.norm2 = nn.InstanceNorm2d(outchannel)

    def forward(self, x):
        source = x
        x = F.leaky_relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        source = self.norm2(self.conv2(source))
        return F.leaky_relu(x + source)


class LocalAttention(nn.Module):
    def __init__(self, img_channel, inchannel):
        super(LocalAttention, self).__init__()
        img_outchannel = [64, 128, 256]
        outchannel = [32, 64, 128, 256]
        flow_channel = 3
        final_outchannel = 3

        # encode source image
        self.convdown_img_1 = ConvDown(img_channel, img_outchannel[0])
        self.convdown_img_2 = ConvDown(img_outchannel[0], img_outchannel[1])
        # self.convdown_img_3 = ConvDown(img_outchannel[1], img_outchannel[2])

        # encode image and pose
        self.convdown_1 = ConvDown(inchannel, outchannel[0])
        self.convdown_2 = ConvDown(outchannel[0], outchannel[1])
        self.convdown_3 = ConvDown(outchannel[1], outchannel[2])
        self.convdown_4 = ConvDown(outchannel[2], outchannel[3])
        self.convdown_5 = ConvDown(outchannel[3], outchannel[3])
        self.resblockup1 = ResBlockUp(outchannel[3], outchannel[3])
        self.resblockup2 = ResBlockUp(outchannel[3], outchannel[2])
        self.resblockupx = ResBlockUp(outchannel[2], outchannel[1])

        self.conv1 = nn.Conv2d(outchannel[1],
                               flow_channel,
                               kernel_size=3,
                               padding=1)
        self.softmax_weight = torch.nn.Softmax(dim=0)
        # decode
        self.resblock1 = ResBlock(outchannel[3], outchannel[3])
        self.resblockup3 = ResBlockUp(outchannel[3], outchannel[2])
        self.resblock2 = ResBlock(outchannel[2], outchannel[2])
        self.resblockup4 = ResBlockUp(outchannel[2], outchannel[1])
        self.resblock3 = ResBlock(outchannel[1], outchannel[1])
        self.resblockup5 = ResBlockUp(outchannel[1], outchannel[0])
        self.conv2 = nn.Conv2d(outchannel[0],
                               final_outchannel,
                               kernel_size=3,
                               padding=1)

    def forward(self, source_imgs, img_and_poses):
        x_total = []
        outputs_flow_x = []
        outputs_flow_y = []
        weight = []
        for source_img, img_and_pose in zip(source_imgs, img_and_poses):
            # encode source image
            source_feature = self.convdown_img_1(source_img)
            source_feature = self.convdown_img_2(source_feature)
            # source_feature = self.convdown_img_3(source_feature)
            # flow
            x1 = self.convdown_1(img_and_pose)
            x1 = self.convdown_2(x1)
            x1 = self.convdown_3(x1)
            x1 = self.convdown_4(x1)
            x2 = self.convdown_5(x1)
            x2 = self.resblockup1(x2)
            x = x1 + x2
            x = self.resblockup2(x)
            x = self.resblockupx(x)
            flow = self.conv1(x)
            flow_x = F.tanh(torch.unsqueeze(flow[:, 0, ...], 1))
            flow_y = F.tanh(torch.unsqueeze(flow[:, 1, ...], 1))
            weight.append(torch.unsqueeze(flow[:, 2, ...], 1))
            print(source_feature.shape)
            print(flow_x.shape)

            x = bilinear_sampler(source_feature, flow_x, flow_y)
            print(x.shape)
            # --------------------add
            resize = transforms.Resize([40, 40])
            # visual_flow_x = resize(flow_x)
            # visual_flow_y = resize(flow_y)
            source_img = resize(source_img)
            visual_img = bilinear_sampler(source_img, flow_x, flow_y)
            # --------------------over
            x_total.append(x)

            outputs_flow_x.append(flow_x.detach())
            outputs_flow_y.append(flow_y.detach())

        ww = torch.stack(weight, dim=0)  # k,b,1,h,w
        ff = torch.stack(x_total, dim=0)  # k,b,c,h,w

        ww = self.softmax_weight(ww)  # k,b,1,h,w
        # print(ww.shape)
        tttt = ww  #torch.sigmoid(ww)
        ff = torch.mul(ww, ff)  # k,b,c,h,w
        ff = torch.sum(ff, dim=0)  # b,c,h,w

        # decode
        # x = self.resblock1(x_total)
        # x = self.resblockup3(x)
        # x = self.resblock2(x)
        # x = self.resblockup4(x)
        # x = self.resblock3(x)
        # x = self.resblockup5(x)
        # x = self.conv2(x)

        return ff, visual_img, outputs_flow_y, tttt
