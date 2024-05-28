import torch
import torch.nn as nn
import numpy as np

from xmuda.models.ft_fz_clip.seg_clip import SegCLIP
from xmuda.models.spconv_unet_v1m1_base import SpUNetBase


class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d,
                 ):
        super(Net2DSeg, self).__init__()

        # 2D network
        if backbone_2d == 'ViT-B-16' or backbone_2d == 'ViT-L-14':
            self.net_2d = SegCLIP(backbone_2d)
            feat_channels = 64
        else:
            raise NotImplementedError('2D backbone {} not supported'.format(backbone_2d))

        self.linear = nn.Linear(feat_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(feat_channels, num_classes)

    def forward(self, data_batch):
        # (batch_size, 3, H, W)
        img = data_batch['img']
        img_indices = data_batch['img_indices']

        x = self.net_2d(img)

        # 2D-3D feature lifting
        img_feats = []
        num_points = []
        for i in range(x.shape[0]):
            proj_feats = x.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]]
            img_feats.append(proj_feats)

            num_point = proj_feats.shape[0]
            num_points.append(num_point)
        img_feats = torch.cat(img_feats, 0)

        x = self.linear(img_feats)

        preds = {
            'feats': img_feats,
            'seg_logit': x,
            'num_points': num_points,
        }

        if self.dual_head:
            preds['seg_logit2'] = self.linear2(img_feats)

        return preds


class Net3DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d,
                 backbone_3d_kwargs,
                 data_source_type,
                 ):
        super(Net3DSeg, self).__init__()

        # 3D network
        if backbone_3d == 'SCN':
            self.net_3d = SpUNetBase(in_channels=backbone_3d_kwargs.in_channels, num_classes=num_classes)
        else:
            raise NotImplementedError('3D backbone {} not supported'.format(backbone_3d))

        # 2nd segmentation head
        self.dual_head = dual_head

    def forward(self, data_batch):
        inter_feat_3d, _ = self.net_3d.encoder_forward(data_batch['x'])
        # inter_feat_3d中的参数会被pop掉
        x1, x2, _ = self.net_3d.decoder_forward(inter_feat_3d)

        preds = {
            'seg_logit': x1,
        }

        if self.dual_head:
            preds['seg_logit2'] = x2

        return preds


if __name__ == '__main__':
    test_Net2DSeg()
    test_Net3DSeg()
