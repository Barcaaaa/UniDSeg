import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import Upsample, resize
from mmcv.cnn import ConvModule

from xmuda.models.ft_fz_clip.clip_vis_model import CLIPVisionTransformer


class FPNHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, in_channels):
        super(FPNHead, self).__init__(input_transform='multiple_select', in_channels=in_channels,
                                      channels=64, num_classes=6, in_index=[0, 1, 2, 3])
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        return output


class SegCLIP(nn.Module):
    def __init__(self, backbone_2d=None):
        super(SegCLIP, self).__init__()

        # ----------------------------------------------------------------------------- #
        # Encoder
        # ----------------------------------------------------------------------------- #
        # 这里的input_resolution为CLIP模型预训练的图像尺寸，forward是会根据输入尺寸对位置编码进行插值处理
        self.backbone_2d = backbone_2d
        if backbone_2d == 'ViT-B-16':
            pretrained_dir = '/data/user1/code/UniDSeg/pretrained/clip/ViT-B-16.pt'
            self.width = 768
            self.image_backbone = CLIPVisionTransformer(input_resolution=640, patch_size=16, width=self.width, layers=12,
                                                        heads=12, output_dim=512, drop_path_rate=0.1, get_embeddings=False,
                                                        out_indices=[3, 5, 7, 11], pretrained=pretrained_dir)
        elif backbone_2d == 'ViT-L-14':
            pretrained_dir = '/data/user1/code/UniDSeg/pretrained/clip/ViT-L-14.pt'
            self.width = 1024
            self.image_backbone = CLIPVisionTransformer(input_resolution=640, patch_size=14, width=self.width, layers=24,
                                                        heads=16, output_dim=512, drop_path_rate=0.1, get_embeddings=False,
                                                        out_indices=[7, 11, 15, 23], pretrained=pretrained_dir)

        # ----------------------------------------------------------------------------- #
        # Decoder
        # ----------------------------------------------------------------------------- #
        self.fpn_head = FPNHead(feature_strides=[4, 8, 16, 32],
                                in_channels=[self.width, self.width, self.width, self.width])
        # TODO: Mask2former maybe get higher performance

    def forward(self, x):
        # pad input to be divisible by 16 = 2 ** 4
        h, w = x.shape[2], x.shape[3]
        min_size = 32
        pad_h = int((h + min_size - 1) / min_size) * min_size - h
        pad_w = int((w + min_size - 1) / min_size) * min_size - w
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [0, pad_w, 0, pad_h])

        # ----------------------------------------------------------------------------- #
        # Encoder
        # ----------------------------------------------------------------------------- #
        inter_features = self.image_backbone(x)

        # ----------------------------------------------------------------------------- #
        # Decoder
        # ----------------------------------------------------------------------------- #
        x_final = self.fpn_head(inter_features)
        x_final = F.interpolate(x_final, (h, w), mode='bilinear', align_corners=True)

        return x_final


if __name__ == '__main__':
    test()
