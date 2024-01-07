import torch
import torch.nn.functional as F


def get_pose_encode(source_imgs, source_poses, target_pose):
    b, _, h, w = source_imgs[0].shape
    encode_all = []

    for i in range(len(source_imgs)):
        _ = target_pose - source_poses[i]
        _ = torch.unsqueeze(_, -1)
        _ = _.repeat(1, 1, h, w)

        encode_all.append(torch.cat([source_imgs[i], _], 1))
    return encode_all


def get_azim_elev_from_poses(poses):
    b, _, _ = poses.shape

    azim = torch.zeros((b, 1)).type_as(poses)
    elev = torch.zeros((b, 1)).type_as(poses)

    for i in range(b):
        azim[i] = torch.nonzero(poses[i, :, 0][0:36])[0][0]
        elev[i] = torch.nonzero(poses[i, :, 0][36:39])[0][0]

    return azim, elev


def bilinear_sampler(source, flow_x, flow_y):
    flow_field = 2 * torch.cat([flow_x, flow_y], 1)

    b, c, h, w = source.shape
    x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source) / (w - 1)
    y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source) / (h - 1)
    grid = torch.stack([x, y], dim=0)
    grid = 2 * grid - 1

    grid = (grid + flow_field).permute(0, 2, 3, 1)
    input_sample = F.grid_sample(source, grid, padding_mode='border')

    return input_sample