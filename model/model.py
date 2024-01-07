import torch.nn as nn
import torch

from base import BaseModel

from model.module_discriminator import DiscriminatorModule
from model.module_gated import GatedModule
from model.module_local_attention import LocalAttention
from model.module_flow import FlowModule
from model.module_tbn import TBN
from model.module_common import bilinear_sampler, get_pose_encode, get_azim_elev_from_poses
from model.loss_models import PatchImageDiscriminator


class FlowModuleProxy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.FlowModule = FlowModule(*args, **kwargs)

    def forward(self, source_img, source_pose, target_pose):
        img_pose_encode = get_pose_encode([source_img], [source_pose],
                                          target_pose)[0]
        _ = self.FlowModule(img_pose_encode)

        output_flow_field_x = torch.unsqueeze(_[:, 0, ...], 1)
        output_flow_field_y = torch.unsqueeze(_[:, 1, ...], 1)

        output_flow = bilinear_sampler(source_img, output_flow_field_x,
                                       output_flow_field_y)

        return output_flow_field_x, output_flow_field_y, output_flow


class FlowModule2DProxy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.FlowModule2D = LocalAttention(*args, **kwargs)

    def forward(self, source_imgs, source_poses, target_pose):
        img_pose_encodes = []

        for source_img, source_pose in zip(source_imgs, source_poses):
            img_pose_encodes.append(
                get_pose_encode([source_img], [source_pose], target_pose)[0])

        return self.FlowModule2D(source_imgs, img_pose_encodes) * 2 - 1


def product(w, x, y, z, ww, xx, yy, zz):
    return w * ww - x * xx - y * yy - z * zz, w * xx + x * ww + y * zz - z * yy, w * yy + y * ww - x * zz + z * xx, w * zz + z * ww + x * yy - y * xx


class FlowModule3DProxy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.FlowModule2D = LocalAttention(3, 11)
        self.FlowModule3D = TBN(*args, **kwargs)

    def forward(self, source_imgs, source_poses, target_pose):
        img_pose_encodes = []
        tgt_azim, tgt_elev = get_azim_elev_from_poses(target_pose)
        yy = torch.deg2rad(tgt_azim * 5)
        xx = torch.deg2rad(tgt_elev * 5)
        tw, tx, ty, tz = product(torch.cos(xx), torch.sin(xx), 0, 0,
                                 torch.cos(yy), 0, torch.sin(yy), 0)
        src_azims = []
        src_elevs = []

        for source_img, source_pose in zip(source_imgs, source_poses):
            src_azim, src_elev = get_azim_elev_from_poses(source_pose)
            src_elevs.append(src_elev)
            src_azims.append(src_azim)
            y = torch.deg2rad(src_azim * 5)
            x = torch.deg2rad(src_elev * 5)
            sw, sx, sy, sz = product(torch.cos(x), torch.sin(x), 0, 0,
                                     torch.cos(y), 0, torch.sin(y), 0)
            tmp = torch.cat((sw, sx, sy, sz, tw, tx, ty, tz), 1)
            img_pose_encodes.append(
                torch.cat((source_img, tmp.unsqueeze(2).unsqueeze(2).expand(
                    -1, -1, 160, 160)), 1))

        feature_2d, outputs_flow_x, outputs_flow_y, weight = self.FlowModule2D(
            source_imgs, img_pose_encodes)
        output_flow_3d = self.FlowModule3D(len(source_imgs), source_imgs,
                                           src_azims, src_elevs, tgt_azim,
                                           tgt_elev, feature_2d)
        # output_flow_3d = self.FlowModule3D(1, source_imgs[1], src_azims, src_elevs, tgt_azim, tgt_elev, feature_2d)
        output_flow_3d = output_flow_3d[0] * 2 - 1
        output_flow_3d = torch.nn.functional.interpolate(output_flow_3d,
                                                         size=(256, 256),
                                                         mode="bilinear")
        return output_flow_3d, outputs_flow_x, outputs_flow_y, weight


class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.FlowModule3D = FlowModule3DProxy(3, 3)
        self.DiscriminatorModule = PatchImageDiscriminator(
            n_channels=3, num_intermediate_layers=2)

        self.initialize_weights(self.FlowModule3D.FlowModule2D.modules())
