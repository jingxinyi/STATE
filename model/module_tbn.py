import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import math
from model.module_gated import GatedModule, GatedBlock
# from model.swin_transformer import SwinTransformer


def apply_volume_transform(input_volume, x_offset, y_offset, z_offset):
    num_batch, num_channels, depth, height, width = input_volume.size()

    im_flat = input_volume.permute(1, 0, 2, 3, 4).contiguous().view(num_channels, -1)
    tensor_type = str(im_flat.type())
    x = torch.linspace(0, width - 1, width).repeat(depth, height, 1).type(tensor_type)
    y = torch.linspace(0, height - 1, height).repeat(depth, width, 1).permute(0, 2, 1).type(tensor_type)
    z = torch.linspace(0, depth - 1, depth).repeat(height, width, 1).permute(2, 0, 1).type(tensor_type)

    x = x.contiguous().view(-1).repeat(1, num_batch)
    y = y.contiguous().view(-1).repeat(1, num_batch)
    z = z.contiguous().view(-1).repeat(1, num_batch)

    x = x + x_offset.contiguous().view(-1)
    y = y + y_offset.contiguous().view(-1)
    z = z + z_offset.contiguous().view(-1)

    x = torch.clamp(x, 0.0, width - 1)
    y = torch.clamp(y, 0.0, height - 1)
    z = torch.clamp(z, 0.0, depth - 1)

    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1
    z0 = torch.floor(z)
    z1 = z0 + 1

    x1 = x1.clamp(max=(width - 1))
    y1 = y1.clamp(max=(height - 1))
    z1 = z1.clamp(max=(depth - 1))

    dim3 = width
    dim2 = width * height
    dim1 = width * height * depth

    base = dim1 * torch.arange(num_batch).type(tensor_type)
    base = base.view(-1, 1).repeat(1, depth * height * width).view(-1)

    base_z0 = base + z0.type(tensor_type) * dim2
    base_z1 = base + z1.type(tensor_type) * dim2

    base_y0z0 = base_z0 + y0 * dim3
    base_y0z1 = base_z1 + y0 * dim3

    base_y1z0 = base_z0 + y1 * dim3
    base_y1z1 = base_z1 + y1 * dim3

    idx_lun = base_y0z0 + x0
    idx_luf = base_y0z1 + x0

    idx_run = base_y0z0 + x1
    idx_ruf = base_y0z1 + x1

    idx_ldn = base_y1z0 + x0
    idx_ldf = base_y1z1 + x0

    idx_rdn = base_y1z0 + x1
    idx_rdf = base_y1z1 + x1

    pix_lun = im_flat.gather(1, idx_lun.repeat(num_channels, 1).long())
    pix_luf = im_flat.gather(1, idx_luf.repeat(num_channels, 1).long())

    pix_run = im_flat.gather(1, idx_run.repeat(num_channels, 1).long())
    pix_ruf = im_flat.gather(1, idx_ruf.repeat(num_channels, 1).long())

    pix_ldn = im_flat.gather(1, idx_ldn.repeat(num_channels, 1).long())
    pix_ldf = im_flat.gather(1, idx_ldf.repeat(num_channels, 1).long())

    pix_rdn = im_flat.gather(1, idx_rdn.repeat(num_channels, 1).long())
    pix_rdf = im_flat.gather(1, idx_rdf.repeat(num_channels, 1).long())

    length_l = (x1 - x)
    length_r = (x - x0)

    length_u = (y1 - y)
    length_d = (y - y0)

    length_n = (z1 - z)
    length_f = (z - z0)

    weight_lun = length_l * length_u * length_n
    weight_luf = length_l * length_u * length_f

    weight_run = length_r * length_u * length_n
    weight_ruf = length_r * length_u * length_f

    weight_ldn = length_l * length_d * length_n
    weight_ldf = length_l * length_d * length_f

    weight_rdn = length_r * length_d * length_n
    weight_rdf = length_r * length_d * length_f

    output = weight_lun * pix_lun + weight_luf * pix_luf + \
             weight_run * pix_run + weight_ruf * pix_ruf + \
             weight_ldn * pix_ldn + weight_ldf * pix_ldf + \
             weight_rdn * pix_rdn + weight_rdf * pix_rdf

    return output.view(num_channels, num_batch, depth, height, width).permute(1, 0, 2, 3, 4)


def ResBlock(num_in_layers, num_out_layers, num_blocks, stride, kernel_size=3, is_3d_conv=False):
    layers = [
        ResConv(num_in_layers, num_out_layers, stride, kernel_size=kernel_size, is_3d_conv=is_3d_conv)
    ]

    for i in range(1, num_blocks - 1):
        layers.append(
            ResConv(4 * num_out_layers, num_out_layers, 1, kernel_size=kernel_size, is_3d_conv=is_3d_conv)
        )

    layers.append(
        ResConv(4 * num_out_layers, num_out_layers, 1, kernel_size=kernel_size, is_3d_conv=is_3d_conv)
    )
    return nn.Sequential(*layers)


class Conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size=3, stride=2, is_3d_conv=False, dilation=1,
                 use_normalization=True,
                 use_relu=False):
        super(Conv, self).__init__()
        self.kernel_size = kernel_size
        self.is_3d_conv = is_3d_conv
        self.dilation = dilation
        self.use_normalization = use_normalization
        self.use_relu = use_relu

        if not self.is_3d_conv:
            self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride,
                                       dilation=self.dilation)
            if self.use_normalization:
                self.normalize = nn.BatchNorm2d(num_out_layers)
        else:
            self.conv_base = nn.Conv3d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride,
                                       dilation=self.dilation)
            if self.use_normalization:
                self.normalize = nn.BatchNorm3d(num_out_layers)

    def forward(self, x):
        p = int(np.floor(self.dilation * (self.kernel_size - 1) / 2))
        if not self.is_3d_conv:
            pd = (p, p, p, p)
        else:
            pd = (p, p, p, p, p, p)
        x = self.conv_base(F.pad(x, pd))
        if self.use_normalization:
            x = self.normalize(x)
        if self.use_relu:
            return F.relu(x, inplace=True)
        else:
            return F.elu(x, inplace=True)


class ResConv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride=2, kernel_size=3, is_3d_conv=False):
        super(ResConv, self).__init__()
        self.is_3d_conv = is_3d_conv
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = Conv(num_in_layers, num_out_layers, 1, 1, self.is_3d_conv)
        self.conv2 = Conv(num_out_layers, num_out_layers, kernel_size=kernel_size, stride=stride,
                          is_3d_conv=self.is_3d_conv)
        if not self.is_3d_conv:
            self.conv3 = nn.Conv2d(num_out_layers, 4 * num_out_layers, kernel_size=1, stride=1)
            self.conv4 = nn.Conv2d(num_in_layers, 4 * num_out_layers, kernel_size=1, stride=stride)
            self.normalize = nn.BatchNorm2d(4 * num_out_layers)
        else:
            self.conv3 = nn.Conv3d(num_out_layers, 4 * num_out_layers, kernel_size=1, stride=1)
            self.conv4 = nn.Conv3d(num_in_layers, 4 * num_out_layers, kernel_size=1, stride=stride)
            self.normalize = nn.BatchNorm3d(4 * num_out_layers)

    def forward(self, x):
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)
        shortcut = self.conv4(x)
        return F.elu(self.normalize(x_out + shortcut), inplace=True)


class UpConv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size=3, scale=2, is_3d_conv=False):
        super(UpConv, self).__init__()
        self.is_3d_conv = is_3d_conv
        self.up_nn = nn.Upsample(scale_factor=scale)
        self.conv1 = Conv(num_in_layers, num_out_layers, kernel_size, 1, is_3d_conv=is_3d_conv)

    def forward(self, x):
        x = self.up_nn(x)
        return self.conv1(x)


class OutputConv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers=3, is_3d_conv=False, kernel_size=3):
        super(OutputConv, self).__init__()
        self.is_3d_conv = is_3d_conv
        self.kernel_size = kernel_size
        self.sigmoid = torch.nn.Sigmoid()
        if not self.is_3d_conv:
            self.conv1 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=self.kernel_size, stride=1)
        else:
            self.conv1 = nn.Conv3d(num_in_layers, num_out_layers, kernel_size=self.kernel_size, stride=1)

    def forward(self, x):
        if self.kernel_size > 1:
            p = 1
            if not self.is_3d_conv:
                pd = (p, p, p, p)
            else:
                pd = (p, p, p, p, p, p)
            x = self.conv1(F.pad(x, pd))
        else:
            x = self.conv1(x)
        x = self.sigmoid(x)
        return x

'''

class SelfAttentionBlock(nn.Module):
    def __init__(self):
        super(SelfAttentionBlock, self).__init__()

    def forward(self, q, k, v):
        """
        Args:
            q (torch.tensor): (N, C, H, W)
            k (torch.tensor): (N, ns, C, H, W)
            v (torch.tensor): (N, ns, C, H, W)
        Returns:
            x (torch.tensor): (N, C, H, W)
        """

        alpha = self.query(q, k)            # (N, ns, 1, H, W)
        x = torch.sum(alpha * v, dim=1)     # (N, ns, C, H, W) * (N, ns, 1, H, W)

        return x

    def query(self, q, k):
        """

        Args:
            q (torch.tensor): (N, C, H, W)
            k (torch.tensor): (N, ns, C, H, W)

        Returns:
            alpha (torch.tensor): (N, ns, 1, H, W)
        """
        dk = k.shape[2]
        q = q.permute(0, 2, 3, 1)        # (N, C, H, W) - > (N, H, W, C)
        k = k.permute(0, 3, 4, 1, 2)     # (N, ns, C, H, W) -> (N, H, W, ns, C)
        q.unsqueeze_(-1)
        alpha = torch.matmul(k, q) / math.sqrt(dk)      # (N, H, W, ns, 1)
        alpha = torch.softmax(alpha, dim=-2)            # (N, H, W, ns, 1)
        alpha = alpha.permute(0, 3, 4, 1, 2)            # (N, ns, 1, H, W)

        return alpha

class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, cond_nc):
        super().__init__()

        assert config_text.startswith("spade")
        parsed = re.search("spade(\D+)(\d)x\d", config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "batch":
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError("%s is not a recognized param-free norm type in SPADE"
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(cond_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, condmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # condmap = F.interpolate(condmap, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(condmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out

    def cal_gamma_beta(self, condmap):
        actv = self.mlp_shared(condmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return gamma, beta

class attSPADE(nn.Module):
    def __init__(self, channel, channel_1):
        super().__init__()

        self.att_block = SelfAttentionBlock()
        self.fq = nn.Conv2d(channel, channel, kernel_size=1)
        self.fk1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.fv1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.fk2 = nn.Conv2d(channel_1, channel, kernel_size=1)
        self.fv2 = nn.Conv2d(channel_1, channel, kernel_size=1)
        self.spade = SPADE("spadeinstance3x3", channel, channel)

    def forward(self, f_str, f_tex, feature):

        kk1 = self.fk1(f_str).unsqueeze_(1)
        kk2 = self.fk2(f_tex).unsqueeze_(1)
        K = torch.cat((kk1, kk2), dim=1)
        vv1 = self.fv1(f_str).unsqueeze_(1)
        vv2 = self.fv2(f_tex).unsqueeze_(1)
        V = torch.cat((vv1, vv2), dim=1)

        q = self.fq(feature)

        x = self.att_block(q, K, V)
        x = self.spade(feature, x)
        return x
'''

# add for self-attention
'''
class ScaledDotProductAttention(nn.Module):


    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        q = q.view(2, 8, 1600, 64)
        k = k.view(2, 8, 1600, 64)

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        attn = self.dropout(F.softmax(attn, dim=1))

        output = torch.matmul(attn, v.view(2, 8, 1600, 64))

        return output, attn

class MultiHeadAttention(nn.Module):


    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(2, 3).transpose(1, 2), k.transpose(2, 3).transpose(1, 2), v.transpose(2, 3).transpose(1, 2)

        q, attn = self.attention(q, k, v)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, len_q, -1)
        q = self.fc(q)
        q = self.dropout(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn

class PositionwiseFeedForward(nn.Module):


    def __init__(self, d_in, dropout=0.1):
        super().__init__()
        self.w_1 = Conv(d_in, d_in, kernel_size=1, stride=1) # position-wise
        self.w_2 = Conv(d_in, d_in, kernel_size=1, stride=1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(2, 3).transpose(1, 2)

        residual = x

        x = self.w_2(self.w_1(x))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x.transpose(1, 2).transpose(2, 3))

        return x

class EncoderLayer(nn.Module):


    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
# add over
'''

class TBN(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, vol_dim=40, num_features=800,
                 tensor_type='torch.cuda.FloatTensor'):
        super(TBN, self).__init__()
        self.num_gen_features = 32
        self.encode_feature_scale_factor = 1
        self.num_input_convs = 1
        self.num_res_convs = 2
        self.w_gen_seg3d = 0
        self.num_output_deconvs = 1
        self.input_height = 256
        self.input_width = 256
        self.azim_rotation_angle_increment = 10
        self.elev_rotation_angle_increment = 10

        # add for self-attention
        '''
        self.d_model = 512
        self.d_inner = 2048
        self.d_k = self.d_v = 64
        self.n_head = 8
        self.n_layers = 4
        '''
        # add over

        self.vol_dim = vol_dim
        self.num_features = num_features
        self.tensor_type = tensor_type
        self.num_enc_features = int(self.num_gen_features / self.encode_feature_scale_factor)
        self.num_dec_features = int(self.num_gen_features / self.encode_feature_scale_factor)

        in_layers = num_in_layers
        if 0 < self.num_input_convs:
            init_num_in_layers = num_in_layers
            middle_num_in_layers = self.num_enc_features
            middle_num_out_layers = middle_num_in_layers
            final_num_out_layers = middle_num_in_layers
            in_layers = final_num_out_layers
            in_conv_layers = []

            for idx in range(self.num_input_convs):
                if 0 == idx:
                    conv_in_layers = init_num_in_layers
                    conv_out_layers = middle_num_in_layers
                elif self.num_input_convs - 1 == idx:
                    conv_in_layers = middle_num_in_layers
                    conv_out_layers = middle_num_out_layers
                else:
                    conv_in_layers = middle_num_in_layers
                    conv_out_layers = final_num_out_layers

                in_conv_layers.append(
                    Conv(num_in_layers=conv_in_layers, num_out_layers=conv_out_layers, kernel_size=4, stride=2,
                         is_3d_conv=False, dilation=1, use_normalization=True)
                )

            self.in_conv = nn.Sequential(*in_conv_layers)

        self.conv1_2d_encode = Conv(in_layers, 2 * self.num_enc_features, 7, 2)
        self.conv2_2d_encode = ResBlock(2 * self.num_enc_features, self.num_enc_features, self.num_res_convs, 2)
        self.conv3_2d_encode = ResBlock(4 * self.num_enc_features, 2 * self.num_enc_features, self.num_res_convs, 2)
        self.conv4_2d_encode = ResBlock(8 * self.num_enc_features, 4 * self.num_enc_features, self.num_res_convs, 2)

        self.upconv4_2d_encode = UpConv(16 * self.num_enc_features, 8 * self.num_enc_features, 3, 2)
        self.iconv4_2d_encode = Conv(2 * 8 * self.num_enc_features, 8 * self.num_enc_features, 3, 1)

        self.upconv3_2d_encode = UpConv(8 * self.num_enc_features, 4 * self.num_enc_features, 3, 2)
        self.iconv3_2d_encode = Conv(2 * 4 * self.num_enc_features, 4 * self.num_enc_features, 3, 1)

        self.upconv2_2d_encode = UpConv(4 * self.num_enc_features, self.num_features, 3, 2)
        self.iconv2_2d_encode = Conv(2 * self.num_enc_features + self.num_features, self.num_features, 3, 1)
        
        self.att = torch.nn.Conv3d(20, 1, 1, 1, 0)
        self.softmax_weight = torch.nn.Softmax(dim=0)
        # self.attspade = attSPADE(800, 128)
        # self.GRU_layer = nn.GRU(1600, 1600, 2)
        # self.output_linear = nn.Linear(1600, 1600)
        # self.swintransformer = SwinTransformer(img_size=40, patch_size=4, in_chans=32, window_size=5, depths=[2, 2], num_heads=[3,6])
        # self.conv3 = UpConv(192, 128, 3, 2)
        # self.conv4 = UpConv(128, 128, 3, 2)
        # self.conv5 = UpConv(128, 128, 3, 2)
        # self.grucell = nn.GRUCell(1600, 1600)
        # self.gru_layer1 = nn.GRU(1600, 1600)

        # add for self-attention
        '''
        self.layer_stack = nn.ModuleList([
            EncoderLayer(self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, dropout=0.1)
            for _ in range(self.n_layers)])
        '''
        #add over

        self.src_seg2d = OutputConv(self.num_features, 1) if 0.0 < self.w_gen_seg3d else None

        self.conv2_2d_decode = ResBlock(self.num_features, self.num_dec_features, self.num_res_convs, 2)
        # self.conv2_2d_decode = ResBlock(512, self.num_dec_features, self.num_res_convs, 2)
        self.conv3_2d_decode = ResBlock(4 * self.num_dec_features, 2 * self.num_dec_features, self.num_res_convs, 2)
        self.conv4_2d_decode = ResBlock(8 * self.num_dec_features, 4 * self.num_dec_features, self.num_res_convs, 2)

        self.upconv4_2d_decode = UpConv(16 * self.num_dec_features, 8 * self.num_dec_features, 3, 2)
        self.iconv4_2d_decode = Conv(2 * 8 * self.num_dec_features, 8 * self.num_dec_features, 3, 1)

        self.upconv3_2d_decode = UpConv(8 * self.num_dec_features, 4 * self.num_dec_features, 3, 2)
        self.iconv3_2d_decode = Conv(2 * 4 * self.num_dec_features, 4 * self.num_dec_features, 3, 1)

        self.upconv2_2d_decode = UpConv(4 * self.num_dec_features, 2 * self.num_dec_features, 3, 2)
        self.iconv2_2d_decode = Conv(self.num_features + 2 * self.num_dec_features, 2 * self.num_dec_features, 3, 1)

        self.upconv1_2d_decode = UpConv(2 * self.num_dec_features, self.num_dec_features, 3, 2)
        self.iconv1_2d_decode = Conv(self.num_dec_features, self.num_dec_features, 3, 1)

        if 0 < self.num_output_deconvs:
            deconv_layers = []
            init_num_in_layers = self.num_dec_features
            middle_num_in_layers = self.num_dec_features
            middle_num_out_layers = middle_num_in_layers
            final_num_out_layers = middle_num_in_layers

            for idx in range(self.num_output_deconvs):
                if 0 == idx:
                    conv_in_layers = init_num_in_layers
                    conv_out_layers = middle_num_in_layers
                elif self.num_output_deconvs - 1 == idx:
                    conv_in_layers = middle_num_in_layers
                    conv_out_layers = middle_num_out_layers
                else:
                    conv_in_layers = middle_num_in_layers
                    conv_out_layers = final_num_out_layers
                deconv_layers.append(UpConv(conv_in_layers, conv_out_layers, 3, 2))

            self.deconv = nn.Sequential(*deconv_layers)

        self.gated_conv = GatedModule(32, 3)

        self.output_2d_decode = OutputConv(self.num_dec_features, num_out_layers)

        num_3d_features = int(self.num_features / self.vol_dim)
        self.conv1_3d_encode = Conv(num_3d_features, self.num_enc_features, 3, 1, is_3d_conv=True)
        self.conv2_3d_encode = Conv(self.num_enc_features, num_3d_features, 3, 1, is_3d_conv=True)

        self.conv1_3d_decode = Conv(num_3d_features, self.num_dec_features, 3, 1, is_3d_conv=True)
        self.conv2_3d_decode = Conv(self.num_dec_features, num_3d_features, 3, 1, is_3d_conv=True)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = Conv(num_features+2**7, num_features, 3, 1, is_3d_conv=False)
        # self.conv1 = Conv(num_features+2**7, 512, 3, 1, is_3d_conv=False)
        # self.layer_norm = nn.LayerNorm(512, eps=1e-6)
        # self.conv2 = Conv(num_features, 128, 3, 1, is_3d_conv=False)
        # self.conv111 = GatedBlock(num_features+2**7, 512*2)#
        # self.conv222 = GatedBlock(512, 128*2)
        # self.conv333 = GatedBlock(128, 32*2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)

    def get_flow_fields(self, crnt_transform_mode, final_elev_transform_mode, init_elev_transform_mode,
                        width, height, depth, encoded_3d_vol):
        rotation_angle = self.azim_rotation_angle_increment * crnt_transform_mode
        init_elev_rotation_angle = self.elev_rotation_angle_increment * init_elev_transform_mode
        final_elev_rotation_angle = self.elev_rotation_angle_increment * final_elev_transform_mode

        rotation_radians = (np.pi / 180.0) * rotation_angle.type_as(rotation_angle)
        init_elev_rotation_radians = (np.pi / 180.0) * init_elev_rotation_angle.type_as(rotation_angle)
        final_elev_rotation_radians = (np.pi / 180.0) * final_elev_rotation_angle.type_as(rotation_angle)

        origxpos = torch.linspace(0, width - 1, width).repeat(height, 1).repeat(depth, 1, 1).type_as(
            rotation_angle) - 2.0 * width / 4.0 + 0.5
        origypos = torch.linspace(0, height - 1, height).repeat(width, 1).permute(1, 0).repeat(depth, 1, 1).type_as(
            rotation_angle) - 2.0 * height / 4.0 + 0.5
        origzpos = torch.linspace(0, depth - 1, depth).repeat(height, 1).repeat(width, 1, 1).permute(2, 1, 0).type_as(
            rotation_angle) - 2.0 * depth / 4.0 + 0.5

        num_batch = crnt_transform_mode.shape[0]
        origxpos = origxpos.repeat(num_batch, 1, 1, 1)
        origypos = origypos.repeat(num_batch, 1, 1, 1)
        origzpos = origzpos.repeat(num_batch, 1, 1, 1)

        rotxpos = origxpos
        rotypos = origypos
        rotzpos = origzpos

        xpos = rotxpos
        ypos = rotypos
        zpos = rotzpos

        init_elev_cos_rad = torch.cos(
            -init_elev_rotation_radians).reshape(
            num_batch, 1).repeat(1, depth * height * width).reshape(num_batch, depth, height, width)

        init_elev_sin_rad = torch.sin(-init_elev_rotation_radians).reshape(num_batch, 1).repeat(1,
                                                                                                depth * height * width).reshape(
            num_batch, depth, height, width)
        rotypos = torch.mul(init_elev_cos_rad, ypos) + \
                  torch.mul(-init_elev_sin_rad, zpos)
        rotzpos = torch.mul(init_elev_sin_rad, ypos) + \
                  torch.mul(init_elev_cos_rad, zpos)
        ypos = rotypos
        zpos = rotzpos

        cos_rad = torch.cos(
            rotation_radians).reshape(
            num_batch, 1
        ).repeat(
            1, depth * height * width
        ).reshape(
            num_batch, depth, height, width
        )

        sin_rad = torch.sin(
            rotation_radians
        ).reshape(num_batch, 1).repeat(1, depth * height * width).reshape(
            num_batch, depth, height, width
        )

        rotxpos = torch.mul(cos_rad, xpos) + \
                  torch.mul(sin_rad, zpos)

        rotzpos = torch.mul(-sin_rad, xpos) + \
                  torch.mul(cos_rad, zpos)

        zpos = rotzpos

        elev_cos_rad = torch.cos(
            final_elev_rotation_radians
        ).reshape(
            num_batch, 1
        ).repeat(
            1, depth * height * width
        ).reshape(
            num_batch, depth, height, width
        )

        elev_sin_rad = torch.sin(
            final_elev_rotation_radians
        ).reshape(
            num_batch, 1
        ).repeat(
            1, depth * height * width
        ).reshape(
            num_batch, depth, height, width
        )

        rotypos = torch.mul(elev_cos_rad, ypos) + \
                  torch.mul(-elev_sin_rad, zpos)
        rotzpos = torch.mul(elev_sin_rad, ypos) + \
                  torch.mul(elev_cos_rad, zpos)

        flow_field_x = (rotxpos - origxpos).reshape(
            num_batch, 1, encoded_3d_vol.shape[2],
            encoded_3d_vol.shape[3], encoded_3d_vol.shape[4]
        )
        flow_field_y = (rotypos - origypos).reshape(
            num_batch, 1, encoded_3d_vol.shape[2],
            encoded_3d_vol.shape[3], encoded_3d_vol.shape[4]
        )
        flow_field_z = (rotzpos - origzpos).reshape(
            num_batch, 1, encoded_3d_vol.shape[2],
            encoded_3d_vol.shape[3], encoded_3d_vol.shape[4]
        )

        flow_field_x = flow_field_x.view(-1).type_as(rotation_angle)
        flow_field_y = flow_field_y.view(-1).type_as(rotation_angle)
        flow_field_z = flow_field_z.view(-1).type_as(rotation_angle)

        return flow_field_x, flow_field_y, flow_field_z

    def forward(self, num_inputs_to_use, x, src_azim, src_elev, tgt_azim, tgt_elev, feature_2d):
        final_tensor = None
        gen_src_seg2d_image = []
        
        enc = []
        con = []
        
        for input_idx in range(num_inputs_to_use):
            encoded, gen_src_seg2d, seg_encoded = self.encode(x[input_idx], src_azim[input_idx], src_elev[input_idx],
                                                              tgt_azim, tgt_elev)
            gen_src_seg2d_image.append(gen_src_seg2d)
            
            enc.append(encoded)
            con.append(self.att(encoded))
            
        weight = torch.stack(con, dim=0)
        feature = torch.stack(enc, dim=0)
        weight = self.softmax_weight(weight)
        feature = torch.mul(weight, feature)
        final_tensor = torch.sum(feature, dim=0)

        gen_tgt_rgb_image = self.decode(final_tensor, feature_2d)

        gen_src_seg3d_image = None
        gen_tgt_seg3d_image = None
        final_seg_tensor = None

        return gen_tgt_rgb_image, gen_src_seg3d_image, gen_tgt_seg3d_image, gen_src_seg2d_image, final_seg_tensor

    def encode(self, x, src_azim, src_elev, tgt_azim, tgt_elev):
        src_rgb = x

        src_azim_transform_mode = src_azim
        src_elev_transform_mode = src_elev

        tgt_azim_transform_mode = tgt_azim
        tgt_elev_transform_mode = tgt_elev

        crnt_transform_mode = src_azim_transform_mode - tgt_azim_transform_mode

        x = src_rgb
        if 0 < self.num_input_convs:
            x = self.in_conv(x)

        x1 = self.conv1_2d_encode(x)
        x2 = self.conv2_2d_encode(x1)
        x3 = self.conv3_2d_encode(x2)
        x4 = self.conv4_2d_encode(x3)

        skip1 = x1
        skip2 = x2
        skip3 = x3

        x_out = self.upconv4_2d_encode(x4)
        x_out = torch.cat((x_out, skip3), 1)
        x_out = self.iconv4_2d_encode(x_out)

        x_out = self.upconv3_2d_encode(x_out)
        x_out = torch.cat((x_out, skip2), 1)
        x_out = self.iconv3_2d_encode(x_out)

        x_out = self.upconv2_2d_encode(x_out)
        x_out = torch.cat((x_out, skip1), 1)
        x_out = self.iconv2_2d_encode(x_out)

        upsample_src_seg2d_output = None

        depth = self.vol_dim
        height = x_out.shape[2]
        width = x_out.shape[3]

        x_out = x_out.view(x_out.shape[0],
                           int(x_out.shape[1] / self.vol_dim), self.vol_dim,
                           x_out.shape[2], x_out.shape[3])

        x_out = self.conv1_3d_encode(x_out)
        encoded_3d_vol = self.conv2_3d_encode(x_out)

        flow_field_x, flow_field_y, flow_field_z = self.get_flow_fields(crnt_transform_mode,
                                                                        src_elev_transform_mode,
                                                                        tgt_elev_transform_mode,
                                                                        width, height, depth, encoded_3d_vol)

        transformed_output = apply_volume_transform(encoded_3d_vol, flow_field_x, flow_field_y, flow_field_z)
        seg_transformed_output = None

        return transformed_output, upsample_src_seg2d_output, seg_transformed_output

    def decode(self, final_transformed_output, feature_2d):
        x_out = self.conv1_3d_decode(final_transformed_output)
        x_out = self.conv2_3d_decode(x_out)

        input_2d = x_out.contiguous().view(x_out.shape[0],
                                           x_out.shape[1] * self.vol_dim,
                                           x_out.shape[3], x_out.shape[4])
        '''
        GRU (false)
        input_2d = input_2d.view(input_2d.shape[0], input_2d.shape[1], input_2d.shape[2] * input_2d.shape[3]).transpose(0, 1).contiguous()
        input_2d, _ = self.GRU_layer(input_2d)
        input_2d = self.output_linear(input_2d)
        input_2d = input_2d.contiguous().view(input_2d.shape[0], input_2d.shape[1], 40, 40).transpose(0, 1)
        
        # feature_2d = torch.zero(x_out.shape[0], 128, 40, 40)

        # feature_2d = self.upsample(feature_2d)
        tmp = torch.cat((input_2d, feature_2d), dim=1)
        tmp = self.conv1(tmp)

        tmp = tmp.view(tmp.shape[0], tmp.shape[1], tmp.shape[2] * tmp.shape[3]).transpose(0, 1).contiguous()
        tmp, _ = self.GRU_layer(tmp)
        tmp = self.output_linear(tmp)
        tmp = tmp.contiguous().view(tmp.shape[0], tmp.shape[1], 40, 40).transpose(0, 1)
        '''
        # spade (failed)
        # tmp = self.attspade(input_2d, feature_2d, tmp)

        # input_2d = self.conv111(tmp)
        # input_2d = self.conv222(input_2d)
        # input_2d = self.conv333(input_2d)
        
        # input_2d = torch.zero(feature_2d.shape[0], 800, 40, 40)

        tmp = torch.cat((input_2d, feature_2d), dim=1)
        tmp = self.conv1(tmp)
        # enc_output = self.layer_norm(tmp.transpose(1, 2).transpose(2, 3))
        # enc_output = self.layer_norm(tmp)

        # add for self-attention
        '''
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
        enc_output = enc_output.transpose(2, 3).transpose(1, 2)
        '''

        # add over

        '''
        tmp = self.conv2(tmp)
        tmp = tmp.view(tmp.shape[0], tmp.shape[1], tmp.shape[2] * tmp.shape[3]).transpose(0, 1).contiguous()
        hid = torch.randn(tmp.shape[1], 1600).unsqueeze(0).cuda()
        _, next_hid = self.gru_layer1(tmp, hid)
        input_2d_1 = input_2d.view(input_2d.shape[0], input_2d.shape[1], input_2d.shape[2] * input_2d.shape[3]).transpose(0, 1).contiguous()
        feature_2d = feature_2d.view(feature_2d.shape[0], feature_2d.shape[1], feature_2d.shape[2] * feature_2d.shape[3]).transpose(0, 1).contiguous()
        x11, _ = self.gru_layer1(feature_2d, next_hid)
        x22, _ = self.gru_layer1(input_2d_1, next_hid)
        x11 = x11.contiguous().view(feature_2d.shape[0], feature_2d.shape[1], 40, 40).transpose(0, 1)
        x22 = x22.contiguous().view(input_2d_1.shape[0], input_2d_1.shape[1], 40, 40).transpose(0, 1)
        x = torch.cat((x11, x22), dim=1)
        x = self.conv1(x)
        # tmp = self.conv2(tmp)
        # tmp = self.swintransformer(tmp)
        # tmp = self.conv3(tmp)
        # tmp = self.conv4(tmp)
        # tmp = self.conv5(tmp)
        '''
        # 800
        x2 = self.conv2_2d_decode(tmp)
        x3 = self.conv3_2d_decode(x2)
        x4 = self.conv4_2d_decode(x3)

        skip1 = input_2d
        skip2 = x2
        skip3 = x3

        x_out = self.upconv4_2d_decode(x4)
        x_out = torch.cat((x_out, skip3), 1)
        x_out = self.iconv4_2d_decode(x_out)

        x_out = self.upconv3_2d_decode(x_out)
        x_out = torch.cat((x_out, skip2), 1)
        x_out = self.iconv3_2d_decode(x_out)

        x_out = self.upconv2_2d_decode(x_out)
        x_out = torch.cat((x_out, skip1), 1)
        x_out = self.iconv2_2d_decode(x_out)

        x_out = self.upconv1_2d_decode(x_out)
        x_out = self.iconv1_2d_decode(x_out)

        if 0 < self.num_output_deconvs:
            x_out = self.deconv(x_out)

        # tgt_img = self.gated_conv(x_out)

        tgt_img = self.output_2d_decode(x_out)
        return tgt_img

# just for 2d
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.num_features = 800
        self.num_dec_features = 32
        self.conv = Conv(2**7, 800, 3, 1, is_3d_conv=False)
        self.conv2_2d_decode = ResBlock(self.num_features, self.num_dec_features, 2, 2)
        self.conv3_2d_decode = ResBlock(4 * self.num_dec_features, 2 * self.num_dec_features, 2, 2)
        self.conv4_2d_decode = ResBlock(8 * self.num_dec_features, 4 * self.num_dec_features, 2, 2)

        self.upconv4_2d_decode = UpConv(16 * self.num_dec_features, 8 * self.num_dec_features, 3, 2)
        self.iconv4_2d_decode = Conv(2 * 8 * self.num_dec_features, 8 * self.num_dec_features, 3, 1)

        self.upconv3_2d_decode = UpConv(8 * self.num_dec_features, 4 * self.num_dec_features, 3, 2)
        self.iconv3_2d_decode = Conv(2 * 4 * self.num_dec_features, 4 * self.num_dec_features, 3, 1)

        self.upconv2_2d_decode = UpConv(4 * self.num_dec_features, 2 * self.num_dec_features, 3, 2)
        self.iconv2_2d_decode = Conv(self.num_features + 2 * self.num_dec_features, 2 * self.num_dec_features, 3, 1)

        self.upconv1_2d_decode = UpConv(2 * self.num_dec_features, self.num_dec_features, 3, 2)
        self.iconv1_2d_decode = Conv(self.num_dec_features, self.num_dec_features, 3, 1)
        self.deconv = UpConv(32, 32, 3, 2)
        self.output_2d_decode = OutputConv(self.num_dec_features, 3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)


    def decode(self, feature):
        input_2d = self.conv(feature)

        # 800
        x2 = self.conv2_2d_decode(input_2d)
        x3 = self.conv3_2d_decode(x2)
        x4 = self.conv4_2d_decode(x3)

        skip1 = input_2d
        skip2 = x2
        skip3 = x3

        x_out = self.upconv4_2d_decode(x4)
        x_out = torch.cat((x_out, skip3), 1)
        x_out = self.iconv4_2d_decode(x_out)

        x_out = self.upconv3_2d_decode(x_out)
        x_out = torch.cat((x_out, skip2), 1)
        x_out = self.iconv3_2d_decode(x_out)

        x_out = self.upconv2_2d_decode(x_out)
        x_out = torch.cat((x_out, skip1), 1)
        x_out = self.iconv2_2d_decode(x_out)

        x_out = self.upconv1_2d_decode(x_out)
        x_out = self.iconv1_2d_decode(x_out)

        x_out = self.deconv(x_out)

        tgt_img = self.output_2d_decode(x_out)
        return tgt_img

    def forward(self, feature2d):
        return self.decode(feature2d)