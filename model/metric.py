import pytorch_msssim
import torch
import pytorch_ssim
import torch.nn as nn

pyssim_loss_module = pytorch_ssim.SSIM(window_size=11)


def grayscale_transform(x):
    return ((x[:, 0, :, :] + x[:, 1, :, :] + x[:, 2, :, :]) / 3.0).unsqueeze(1)


def metric_l1(x, y):
    # return torch.mean(torch.abs(x - y))
    return nn.functional.l1_loss(x, y)


def metric_tbn_ssim(x, y):
    x = (x + 1) / 2
    y = (y + 1) / 2

    return pyssim_loss_module(grayscale_transform(x), grayscale_transform(y))


def metric_ssim(x, y):
    x = (x + 1) / 2
    y = (y + 1) / 2

    return pytorch_msssim.ssim(x, y, data_range=1, size_average=True)


def metric_msssim(x, y):
    x = (x + 1) / 2
    y = (y + 1) / 2

    return pytorch_msssim.ms_ssim(x, y, data_range=1, size_average=True)
