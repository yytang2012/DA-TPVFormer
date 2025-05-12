import numpy as np
import torch
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmengine.registry import MODELS
from torch import nn
from torch.nn.init import normal_

from .cross_view_hybrid_attention import TPVCrossViewHybridAttention
from .image_cross_attention import TPVMSDeformableAttention3D
import math

@MODELS.register_module()
class TPVFormerEncoder(TransformerLayerSequence):

    def __init__(self,
                 tpv_h=200,
                 tpv_w=200,
                 tpv_z=16,
                 pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                 num_feature_levels=4,
                 num_cams=6,
                 embed_dims=256,
                 num_points_in_pillar=[4, 32, 32],
                 num_points_in_pillar_cross_view=[32, 32, 32],
                 num_layers=5,
                 transformerlayers=None,
                 positional_encoding=None,
                 return_intermediate=False):
        super().__init__(transformerlayers, num_layers)

        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.pc_range = pc_range
        self.real_w = pc_range[3] - pc_range[0]
        self.real_h = pc_range[4] - pc_range[1]
        self.real_z = pc_range[5] - pc_range[2]

        self.level_embeds = nn.Parameter(
            torch.Tensor(num_feature_levels, embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(num_cams, embed_dims))
        self.tpv_embedding_hw = nn.Embedding(tpv_h * tpv_w, embed_dims)
        self.tpv_embedding_zh = nn.Embedding(tpv_z * tpv_h, embed_dims)
        self.tpv_embedding_wz = nn.Embedding(tpv_w * tpv_z, embed_dims)

        ref_3d_hw = self.get_reference_points(tpv_h, tpv_w, self.real_z,
                                              num_points_in_pillar[0])
        ref_3d_zh = self.get_reference_points(tpv_z, tpv_h, self.real_w,
                                              num_points_in_pillar[1])
        ref_3d_zh = ref_3d_zh.permute(3, 0, 1, 2)[[2, 0, 1]]  # change to x,y,z
        ref_3d_zh = ref_3d_zh.permute(1, 2, 3, 0)
        ref_3d_wz = self.get_reference_points(tpv_w, tpv_z, self.real_h,
                                              num_points_in_pillar[2])
        ref_3d_wz = ref_3d_wz.permute(3, 0, 1, 2)[[1, 2, 0]]  # change to x,y,z
        ref_3d_wz = ref_3d_wz.permute(1, 2, 3, 0)
        self.register_buffer('ref_3d_hw', ref_3d_hw)
        self.register_buffer('ref_3d_zh', ref_3d_zh)
        self.register_buffer('ref_3d_wz', ref_3d_wz)

        cross_view_ref_points = self.get_cross_view_ref_points(
            tpv_h, tpv_w, tpv_z, num_points_in_pillar_cross_view)
        self.register_buffer('cross_view_ref_points', cross_view_ref_points)

        # positional encoding
        self.positional_encoding = MODELS.build(positional_encoding)
        self.return_intermediate = return_intermediate

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, TPVMSDeformableAttention3D) or isinstance(
                    m, TPVCrossViewHybridAttention):
                m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    @staticmethod
    def get_cross_view_ref_points(tpv_h, tpv_w, tpv_z, num_points_in_pillar):
        # ref points generating target: (#query)hw+zh+wz, (#level)3, #p, 2
        # generate points for hw and level 1
        h_ranges = torch.linspace(0.5, tpv_h - 0.5, tpv_h) / tpv_h
        w_ranges = torch.linspace(0.5, tpv_w - 0.5, tpv_w) / tpv_w
        h_ranges = h_ranges.unsqueeze(-1).expand(-1, tpv_w).flatten()
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_h, -1).flatten()
        hw_hw = torch.stack([w_ranges, h_ranges], dim=-1)  # hw, 2
        hw_hw = hw_hw.unsqueeze(1).expand(-1, num_points_in_pillar[2],
                                          -1)  # hw, #p, 2
        # generate points for hw and level 2
        z_ranges = torch.linspace(0.5, tpv_z - 0.5,
                                  num_points_in_pillar[2]) / tpv_z  # #p
        z_ranges = z_ranges.unsqueeze(0).expand(tpv_h * tpv_w, -1)  # hw, #p
        h_ranges = torch.linspace(0.5, tpv_h - 0.5, tpv_h) / tpv_h
        h_ranges = h_ranges.reshape(-1, 1, 1).expand(
            -1, tpv_w, num_points_in_pillar[2]).flatten(0, 1)
        hw_zh = torch.stack([h_ranges, z_ranges], dim=-1)  # hw, #p, 2
        # generate points for hw and level 3
        z_ranges = torch.linspace(0.5, tpv_z - 0.5,
                                  num_points_in_pillar[2]) / tpv_z  # #p
        z_ranges = z_ranges.unsqueeze(0).expand(tpv_h * tpv_w, -1)  # hw, #p
        w_ranges = torch.linspace(0.5, tpv_w - 0.5, tpv_w) / tpv_w
        w_ranges = w_ranges.reshape(1, -1, 1).expand(
            tpv_h, -1, num_points_in_pillar[2]).flatten(0, 1)
        hw_wz = torch.stack([z_ranges, w_ranges], dim=-1)  # hw, #p, 2

        # generate points for zh and level 1
        w_ranges = torch.linspace(0.5, tpv_w - 0.5,
                                  num_points_in_pillar[1]) / tpv_w
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_z * tpv_h, -1)
        h_ranges = torch.linspace(0.5, tpv_h - 0.5, tpv_h) / tpv_h
        h_ranges = h_ranges.reshape(1, -1, 1).expand(
            tpv_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        zh_hw = torch.stack([w_ranges, h_ranges], dim=-1)
        # generate points for zh and level 2
        z_ranges = torch.linspace(0.5, tpv_z - 0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(
            -1, tpv_h, num_points_in_pillar[1]).flatten(0, 1)
        h_ranges = torch.linspace(0.5, tpv_h - 0.5, tpv_h) / tpv_h
        h_ranges = h_ranges.reshape(1, -1, 1).expand(
            tpv_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        zh_zh = torch.stack([h_ranges, z_ranges], dim=-1)  # zh, #p, 2
        # generate points for zh and level 3
        w_ranges = torch.linspace(0.5, tpv_w - 0.5,
                                  num_points_in_pillar[1]) / tpv_w
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_z * tpv_h, -1)
        z_ranges = torch.linspace(0.5, tpv_z - 0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(
            -1, tpv_h, num_points_in_pillar[1]).flatten(0, 1)
        zh_wz = torch.stack([z_ranges, w_ranges], dim=-1)

        # generate points for wz and level 1
        h_ranges = torch.linspace(0.5, tpv_h - 0.5,
                                  num_points_in_pillar[0]) / tpv_h
        h_ranges = h_ranges.unsqueeze(0).expand(tpv_w * tpv_z, -1)
        w_ranges = torch.linspace(0.5, tpv_w - 0.5, tpv_w) / tpv_w
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(
            -1, tpv_z, num_points_in_pillar[0]).flatten(0, 1)
        wz_hw = torch.stack([w_ranges, h_ranges], dim=-1)
        # generate points for wz and level 2
        h_ranges = torch.linspace(0.5, tpv_h - 0.5,
                                  num_points_in_pillar[0]) / tpv_h
        h_ranges = h_ranges.unsqueeze(0).expand(tpv_w * tpv_z, -1)
        z_ranges = torch.linspace(0.5, tpv_z - 0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(
            tpv_w, -1, num_points_in_pillar[0]).flatten(0, 1)
        wz_zh = torch.stack([h_ranges, z_ranges], dim=-1)
        # generate points for wz and level 3
        w_ranges = torch.linspace(0.5, tpv_w - 0.5, tpv_w) / tpv_w
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(
            -1, tpv_z, num_points_in_pillar[0]).flatten(0, 1)
        z_ranges = torch.linspace(0.5, tpv_z - 0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(
            tpv_w, -1, num_points_in_pillar[0]).flatten(0, 1)
        wz_wz = torch.stack([z_ranges, w_ranges], dim=-1)

        reference_points = torch.cat([
            torch.stack([hw_hw, hw_zh, hw_wz], dim=1),
            torch.stack([zh_hw, zh_zh, zh_wz], dim=1),
            torch.stack([wz_hw, wz_zh, wz_wz], dim=1)
        ],
                                     dim=0)  # hw+zh+wz, 3, #p, 2

        return reference_points

    @staticmethod
    def get_reference_points(H,
                             W,
                             Z=8,
                             num_points_in_pillar=4,
                             dim='3d',
                             bs=1,
                             device='cuda',
                             dtype=torch.float):
        """Get the reference points used in SCA and TSA.

        Args:
            H, W: spatial shape of tpv.
            Z: height of pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        zs = torch.linspace(
            0.5, Z - 0.5, num_points_in_pillar,
            dtype=dtype, device=device).view(-1, 1, 1).expand(
                num_points_in_pillar, H, W) / Z
        xs = torch.linspace(
            0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, -1).expand(
                num_points_in_pillar, H, W) / W
        ys = torch.linspace(
            0.5, H - 0.5, H, dtype=dtype, device=device).view(1, -1, 1).expand(
                num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
        return ref_3d

    def point_sampling(self, reference_points, pc_range, batch_data_smaples):

        lidar2img = []
        for data_sample in batch_data_smaples:
            lidar2img.append(data_sample.lidar2img)
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(
            1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4,
                                   4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32),
            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        tpv_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= data_sample.batch_input_shape[1]
        reference_points_cam[..., 1] /= data_sample.batch_input_shape[0]

        tpv_mask = (
            tpv_mask & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0))

        tpv_mask = torch.nan_to_num(tpv_mask)

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        tpv_mask = tpv_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, tpv_mask

    def pre_sampling(self, pc_range, pc_range_h):
        h_range = pc_range[3] - pc_range[0]
        w_range = pc_range[4] - pc_range[1]
        z_range = pc_range[5] - pc_range[2]
        voxel_size_h = h_range / self.tpv_h
        voxel_size_w = w_range / self.tpv_w
        voxel_size_z = z_range / self.tpv_z
        left_range = [pc_range[0], pc_range[1], pc_range[2]]

        start_h = math.floor((pc_range_h[0] - left_range[0]) / voxel_size_h)
        end_h = math.floor((pc_range_h[3] - left_range[0]) / voxel_size_h)
        start_w = math.floor((pc_range_h[1] - left_range[1]) / voxel_size_w)
        end_w = math.floor((pc_range_h[4] - left_range[1]) / voxel_size_w)
        start_z = math.floor((pc_range_h[2] - left_range[2]) / voxel_size_z)
        end_z = math.floor((pc_range_h[5] - left_range[2]) / voxel_size_z)

        # h_1 = (pc_range_h[0] - left_range[0]) / voxel_size_h
        # frac = math.ceil(h_1) - h_1
        # if frac >= 0.001:
        #     start_h = math.floor(h_1)
        # else:
        #     start_h = math.ceil(h_1)
        #
        # h_2 = (pc_range_h[3] - left_range[0]) / voxel_size_h
        # frac = h_2 - math.floor(h_2)
        # if frac >= 0.001:
        #     end_h = math.ceil(h_2)
        # else:
        #     end_h = math.floor(h_2)
        #
        # w_1 = (pc_range_h[1] - left_range[1]) / voxel_size_w
        # frac = math.ceil(w_1) - w_1
        # if frac >= 0.001:
        #     start_w = math.floor(w_1)
        # else:
        #     start_w = math.ceil(w_1)
        #
        # w_2 = (pc_range_h[4] - left_range[1]) / voxel_size_w
        # frac = w_2 - math.floor(w_2)
        # if frac >= 0.001:
        #     end_w = math.ceil(w_2)
        # else:
        #     end_w = math.floor(w_2)
        #
        # z_1 = (pc_range_h[2] - left_range[2]) / voxel_size_z
        # frac = math.ceil(z_1) - z_1
        # if frac >= 0.001:
        #     start_z = math.floor(z_1)
        # else:
        #     start_z = math.ceil(z_1)
        #
        # z_2 = (pc_range_h[5] - left_range[2]) / voxel_size_z
        # frac = z_2 - math.floor(z_2)
        # if frac >= 0.001:
        #     end_z = math.ceil(z_2)
        # else:
        #     end_z = math.floor(z_2)

        return start_h, end_h, start_w, end_w, start_z, end_z

    def forward(self, mlvl_feats, batch_data_samples, mode='low', tpv_xcy=None, pc_range_h=None):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        """
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device

        # tpv queries and pos embeds
        tpv_queries_hw = self.tpv_embedding_hw.weight.to(dtype)
        tpv_queries_zh = self.tpv_embedding_zh.weight.to(dtype)
        tpv_queries_wz = self.tpv_embedding_wz.weight.to(dtype)
        tpv_queries_hw = tpv_queries_hw.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_zh = tpv_queries_zh.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_wz = tpv_queries_wz.unsqueeze(0).repeat(bs, 1, 1)

        e_dim = tpv_queries_hw.shape[2]
        if mode == 'low':
            pc_range_h = pc_range_h
            pc_range = self.pc_range
            start_h, end_h, start_w, end_w, start_z, end_z = self.pre_sampling(pc_range, pc_range_h)
            len_h = end_h - start_h
            len_w = end_w - start_w
            len_z = end_z - start_z
            tpv_queries_hw = tpv_queries_hw.reshape(1, self.tpv_h, self.tpv_w, e_dim)
            tpv_queries_zh = tpv_queries_zh.reshape(1, self.tpv_z, self.tpv_h, e_dim)
            tpv_queries_wz = tpv_queries_wz.reshape(1, self.tpv_w, self.tpv_z, e_dim)
            tpv_queries_hw[:, start_h: end_h, start_w: end_w, :] *= 0.2
            tpv_queries_zh[:, start_z: end_z, start_h: end_h, :] *= 0.2
            tpv_queries_wz[:, start_w: end_w, start_z: end_z, :] *= 0.2
            tpv_xcy_hw = tpv_xcy[0].reshape(1, len_h, len_w, e_dim)
            tpv_xcy_zh = tpv_xcy[1].reshape(1, len_z, len_h, e_dim)
            tpv_xcy_wz = tpv_xcy[2].reshape(1, len_w, len_z, e_dim)
            tpv_queries_hw[:, start_h: end_h, start_w: end_w, :] += (tpv_xcy_hw * 0.8)
            tpv_queries_zh[:, start_z: end_z, start_h: end_h, :] += (tpv_xcy_zh * 0.8)
            tpv_queries_wz[:, start_w: end_w, start_z: end_z, :] += (tpv_xcy_wz * 0.8)
            # tpv_queries_hw[:, self.tpv_h // 4: self.tpv_h * 3 // 4, self.tpv_w // 4: self.tpv_w * 3 // 4, :] *= 0.2
            # tpv_queries_zh[:, self.tpv_z // 4: self.tpv_z * 3 // 4, self.tpv_h // 4: self.tpv_h * 3 // 4, :] *= 0.2
            # tpv_queries_wz[:, self.tpv_w // 4: self.tpv_w * 3 // 4, self.tpv_z // 4: self.tpv_z * 3 // 4, :] *= 0.2
            # tpv_xcy_hw = tpv_xcy[0].reshape(1, self.tpv_h // 2, self.tpv_w // 2, e_dim)
            # tpv_xcy_zh = tpv_xcy[1].reshape(1, self.tpv_z // 2, self.tpv_h // 2, e_dim)
            # tpv_xcy_wz = tpv_xcy[2].reshape(1, self.tpv_w // 2, self.tpv_z // 2, e_dim)
            # tpv_queries_hw[:, self.tpv_h // 4: self.tpv_h * 3 // 4, self.tpv_w // 4: self.tpv_w * 3 // 4, :] += (tpv_xcy_hw * 0.8)
            # tpv_queries_zh[:, self.tpv_z // 4: self.tpv_z * 3 // 4, self.tpv_h // 4: self.tpv_h * 3 // 4, :] += (tpv_xcy_zh * 0.8)
            # tpv_queries_wz[:, self.tpv_w // 4: self.tpv_w * 3 // 4, self.tpv_z // 4: self.tpv_z * 3 // 4, :] += (tpv_xcy_wz * 0.8)
            tpv_queries_hw = tpv_queries_hw.reshape(1, self.tpv_h * self.tpv_w, e_dim)
            tpv_queries_zh = tpv_queries_zh.reshape(1, self.tpv_z * self.tpv_h, e_dim)
            tpv_queries_wz = tpv_queries_wz.reshape(1, self.tpv_w * self.tpv_z, e_dim)
        tpv_query = [tpv_queries_hw, tpv_queries_zh, tpv_queries_wz]

        tpv_pos_hw = self.positional_encoding(bs, device, 'z')
        tpv_pos_zh = self.positional_encoding(bs, device, 'w')
        tpv_pos_wz = self.positional_encoding(bs, device, 'h')
        tpv_pos = [tpv_pos_hw, tpv_pos_zh, tpv_pos_wz]

        # flatten image features of different scales
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # num_cam, bs, hw, c
            feat = feat + self.cams_embeds[:, None, None, :].to(dtype)
            feat = feat + self.level_embeds[None, None,
                                            lvl:lvl + 1, :].to(dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)  # num_cam, bs, hw++, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        reference_points_cams, tpv_masks = [], []
        ref_3ds = [self.ref_3d_hw, self.ref_3d_zh, self.ref_3d_wz]
        for ref_3d in ref_3ds:
            reference_points_cam, tpv_mask = self.point_sampling(
                ref_3d, self.pc_range,
                batch_data_samples)  # num_cam, bs, hw++, #p, 2
            reference_points_cams.append(reference_points_cam)
            tpv_masks.append(tpv_mask)

        ref_cross_view = self.cross_view_ref_points.clone().unsqueeze(
            0).expand(bs, -1, -1, -1, -1)

        intermediate = []
        for layer in self.layers:
            output = layer(
                tpv_query,
                feat_flatten,
                feat_flatten,
                tpv_pos=tpv_pos,
                ref_2d=ref_cross_view,
                tpv_h=self.tpv_h,
                tpv_w=self.tpv_w,
                tpv_z=self.tpv_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cams=reference_points_cams,
                tpv_masks=tpv_masks)
            tpv_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output
