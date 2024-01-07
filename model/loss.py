import torch
import pytorch_msssim
from torch import nn
import torchvision.models as models
import sys
import torch.nn.functional as F
from .loss_models import PerceptualVGG19
import pytorch_ssim


mod = sys.modules[__name__]


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


class VGGLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(VGGLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        # content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        # content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return content_loss, style_loss


def loss_l1(x, y):
    return torch.mean(torch.abs(x - y))


def loss_msssim(x, y):
    x = (x + 1) / 2
    y = (y + 1) / 2
    loss = 1 - pytorch_msssim.ms_ssim(x, y, data_range=1, size_average=True)

    return loss


def loss_ssim(x, y):
    x = (x + 1) / 2
    y = (y + 1) / 2
    loss = 1 - pytorch_msssim.ssim(x, y, data_range=1, size_average=True)

    return loss


def loss_vgg(x, y):
    content_loss, style_loss = VGGLoss(x, y)

    return content_loss, style_loss


def loss_clamp(x):
    return torch.mean(F.relu(-x-1)) + torch.mean(F.relu(x-1))


class Loss(nn.modules.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.pyssim_loss_module = pytorch_ssim.SSIM(window_size=11)
        self.vgg_model_path = ''

        self.perception_loss_module = PerceptualVGG19(feature_layers=[0, 5, 10, 15], use_normalization=False,
                                                          path=self.vgg_model_path)
        self.perception_loss_module = self.perception_loss_module.cuda()

        self.l1_loss_module = nn.L1Loss().cuda()

        self.cos = nn.CosineSimilarity(dim=1).cuda()

    def grayscale_transform(self, x):
        return ((x[:, 0, :, :] + x[:, 1, :, :] + x[:, 2, :, :]) / (3.0)).unsqueeze(1)

    def forward(self, input, target):

        _, fake_features = self.perception_loss_module(input)
        _, tgt_features = self.perception_loss_module(target)
        vgg_tgt = ((fake_features - tgt_features) ** 2).mean()

        l1_tgt = self.l1_loss_module(input, target)

        ssim_tgt = self.pyssim_loss_module(self.grayscale_transform(input),
                                           self.grayscale_transform(target))
        cos_tgt = 1 - self.cos(input, target).mean()

        return vgg_tgt, l1_tgt, ssim_tgt, cos_tgt


# VGGLoss = VGGLoss().cuda()
loss_fn = Loss()


def loss(*args, use_msssim=False):
    x = args[0]
    if len(args) == 1:
        return loss_clamp(x)

    y = args[1]

    k = [1, 10, 0.5, 1]

    loss_vgg_result, loss_l1_result, loss_ssim_result, loss_cos_result = loss_fn(x, y)

    # Normalizes SSIM loss to range [0.0, 2.0]
    nonzero_ssim_loss = (loss_ssim_result + 1.0)
    loss_ssim_result = (2.0 - nonzero_ssim_loss)

    loss = loss_l1_result * k[0] + loss_ssim_result * k[1] + loss_vgg_result * k[2] + loss_cos_result * k[3]

    return loss, [v.detach().item() for v in [loss_l1_result, loss_ssim_result, loss_vgg_result]]


def loss_ms(*args):
    return loss(*args, use_msssim=True)
