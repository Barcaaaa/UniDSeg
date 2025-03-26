import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor
import numpy as np

from timm.models.layers import trunc_normal_


class MTP(nn.Module):
    def __init__(self, scale_factor, embed_dim, patch_size, num_layers):
        """
        Args:
        """
        super(MTP, self).__init__()
        self.scale_factor = scale_factor
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_layers = num_layers

        self.embed_img_layer = nn.Linear(self.embed_dim, self.embed_dim//self.scale_factor)
        self.embed_dep_layer = nn.Conv2d(in_channels=3, out_channels=embed_dim // scale_factor,
                                         kernel_size=patch_size, stride=patch_size, bias=False)
        self.embed_fft_layer = nn.Conv2d(in_channels=3, out_channels=embed_dim // scale_factor,
                                         kernel_size=patch_size, stride=patch_size, bias=False)

        for i in range(self.num_layers):
            exclusive_mlp = nn.Sequential(
                nn.Linear(self.embed_dim // self.scale_factor * 3, self.embed_dim // self.scale_factor * 3),
                nn.GELU(),
                nn.Linear(self.embed_dim // self.scale_factor * 3, self.embed_dim // self.scale_factor),
            )
            setattr(self, 'exclusive_mlp_{}'.format(str(i)), exclusive_mlp)
        self.shared_mlp = nn.Linear(self.embed_dim // self.scale_factor, self.embed_dim)

        self.apply(self._init_weights)

        self.freq_nums = 0.25

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_embed(self, x):
        N, C, H, W = x.shape  # [8,768,16,26]
        x = x.reshape(N, C, H*W).permute(0, 2, 1)  # [8,416,768]
        return self.embed_img_layer(x)  # [8,416,48]

    def init_embed_fft(self, x):
        x = self.fft(x, self.freq_nums)
        # x = self.pasta(x)
        x = self.embed_fft_layer(x)  # [8,48,16,26]
        N, C, H, W = x.shape
        return x.reshape(N, C, H*W).permute(0, 2, 1)  # [8,416,48]

    def init_embed_dep(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.embed_dep_layer(x)  # [8,48,16,26]
        N, C, H, W = x.shape
        return x.reshape(N, C, H*W).permute(0, 2, 1)  # [8,416,48]

    def get_prompt(self, feat, feat_dep, feat_fft):
        prompts = []
        for i in range(self.num_layers):
            exclusive_mlp = getattr(self, 'exclusive_mlp_{}'.format(str(i)))
            prompt = exclusive_mlp(torch.cat([feat, feat_dep, feat_fft], dim=-1))  # [8,416,48]
            prompts.append(self.shared_mlp(prompt))  # [8,416,768]
        return prompts # 12*[8,416,768]

    def fft(self, x, rate):
        # the smaller rate, the smoother; the larger rate, the darker
        # rate = 4, 8, 16, 32
        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))  # norm="forward"指定了傅里叶变换的归一化方式
        # mask[fft.float() > self.freq_nums] = 1
        # high pass: 1-mask, low pass: mask
        #fft = fft * (1 - mask)
        fft_low = fft * mask
        fr_low = fft_low.real
        fi_low = fft_low.imag

        fft_hires_low = torch.fft.ifftshift(torch.complex(fr_low, fi_low))
        inv_low = torch.fft.ifft2(fft_hires_low, norm="forward").real

        inv_low = torch.abs(inv_low)

        return inv_low

    # def pasta(self, x):
    #     fft_src = torch.fft.fft2(x, norm="forward")
    #     amp_src, pha_src = torch.abs(fft_src).cpu(), torch.angle(fft_src).cpu()
    #
    #     X, Y = amp_src.shape[2:]
    #     X_range, Y_range = None, None
    #     if X % 2 == 1:
    #         X_range = np.arange(-1 * (X // 2), (X // 2) + 1)
    #     else:
    #         X_range = np.concatenate(
    #             [np.arange(-1 * (X // 2) + 1, 1), np.arange(0, X // 2)]
    #         )
    #
    #     if Y % 2 == 1:
    #         Y_range = np.arange(-1 * (Y // 2), (Y // 2) + 1)
    #     else:
    #         Y_range = np.concatenate(
    #             [np.arange(-1 * (Y // 2) + 1, 1), np.arange(0, Y // 2)]
    #         )
    #
    #     XX, YY = np.meshgrid(Y_range, X_range)
    #
    #     alpha = 3.0
    #     beta = 0.25
    #     k = 2
    #     inv = np.sqrt(np.square(XX) + np.square(YY))
    #     inv *= (1 / inv.max()) * alpha
    #     inv = np.power(inv, k)
    #     inv = np.tile(inv, (3, 1, 1))
    #     inv += beta
    #     prop = np.fft.fftshift(inv, axes=[-2, -1])
    #     amp_src = amp_src * np.random.normal(np.ones(prop.shape), prop)
    #
    #     aug_img = amp_src * torch.exp(1j * pha_src)
    #     inv_pasta = torch.fft.ifft2(aug_img, norm="forward").real
    #     inv_pasta = torch.abs(inv_pasta).float().cuda()
    #
    #     return inv_pasta


class LST(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        low_rank_dim: int = 32,
        token_length: int = 100,
        scale_init: float = 0.001,
    ) -> None:
        super(LST, self).__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.low_rank_dim = low_rank_dim
        self.token_length = token_length
        self.scale_init = scale_init
        self.create_model()

    def create_model(self):
        # 创建一个num_layers层，token长度为token_length，维度为embed_dims的可学习张量
        self.learnable_tokens_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.low_rank_dim])
        )
        self.learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.low_rank_dim, self.embed_dims])
        )

        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_1 = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_2 = nn.Linear(self.embed_dims, self.embed_dims)

        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + (self.embed_dims * self.low_rank_dim) ** 0.5
            )
        )
        nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_2.weight, a=math.sqrt(5))

    def get_tokens(self, layer: int) -> Tensor:
        return self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]

    def forward(self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)  # (HW,B,C)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)

        tokens = self.get_tokens(layer)
        feats = self.tune_query(feats, tokens, layer) # refined feature

        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)  # (B,HW,C)
        return feats

    def tune_query(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        J = torch.einsum("nbc,mc->nbm", feats, tokens)
        J = J * (self.embed_dims**-0.5)
        J = F.softmax(J, dim=-1)

        delta_feat = torch.einsum("nbm,mc->nbc", J[:, :, 1:], self.mlp_1(tokens[1:, :]))
        delta_feat = self.mlp_2(feats + delta_feat)
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat

        return feats


class LSTv1(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        low_rank_dim: int = 32,
        token_length: int = 50,
        scale_init: float = 0.001,
        r1: int = 1,
        r2: int = 128,
    ) -> None:
        super(LSTv1, self).__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.low_rank_dim = low_rank_dim
        self.token_length = token_length
        self.scale_init = scale_init
        self.r1 = r1
        self.r2 = r2
        self.create_model()

    def create_model(self):
        # 创建一个num_layers层，token长度为token_length，维度为embed_dims的可学习张量
        self.learnable_tokens_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.low_rank_dim])
        )
        self.learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.low_rank_dim, self.embed_dims])
        )

        self.down_layer1 = nn.Linear(self.embed_dims, self.r1, bias=False)
        self.up_layer1 = nn.Linear(self.r1, self.embed_dims, bias=False)
        self.down_layer2 = nn.Linear(self.embed_dims, self.r2, bias=False)
        self.up_layer2 = nn.Linear(self.r2, self.embed_dims, bias=False)
        # self.act_func = nn.GELU()

        self.ca = ChannelFilter()

        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_1 = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_2 = nn.Linear(self.embed_dims, self.embed_dims)

        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + (self.embed_dims * self.low_rank_dim) ** 0.5
            )
        )
        nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b.data, -val, val)
        nn.init.normal_(self.down_layer1.weight, std=1 / self.r1 ** 2)
        nn.init.zeros_(self.up_layer1.weight)
        nn.init.normal_(self.down_layer2.weight, std=1 / self.r2 ** 2)
        nn.init.zeros_(self.up_layer2.weight)
        nn.init.kaiming_uniform_(self.mlp_1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_2.weight, a=math.sqrt(5))

    def get_tokens(self, layer: int) -> Tensor:
        return self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]

    def forward(self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)  # (HW,B,C)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)

        tokens = self.get_tokens(layer)
        feats = self.tune_query(feats, tokens, layer) # refined feature

        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)  # (B,HW,C)
        return feats

    def tune_query(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        J = torch.einsum("nbc,mc->nbm", feats, tokens)
        J = J * (self.embed_dims**-0.5)
        J = F.softmax(J, dim=-1)

        delta_feat = torch.einsum("nbm,mc->nbc", J[:, :, 1:], self.mlp_1(tokens[1:, :]))

        ca_feats = self.ca(feats)
        down_up_feat1 = self.up_layer1(self.down_layer1(ca_feats))
        down_up_feat2 = self.up_layer2(self.down_layer2(ca_feats))

        delta_feat = self.mlp_2(feats + delta_feat)
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat + (down_up_feat1 + down_up_feat2)

        return feats


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        output = self.sigmoid(max_result + avg_result)
        return output

class ChannelFilter(nn.Module):
    def __init__(self, channel=768):
        super().__init__()
        self.ca = ChannelAttention(channel)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        output = self.ca(x)
        x = x * output
        x = x.permute(1, 0, 2)
        return x
