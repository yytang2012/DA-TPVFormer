from typing import Optional, Union

from torch import nn

from mmdet3d.models import Base3DSegmentor
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
import copy
import torch
import torch.nn.functional as F
import math
@MODELS.register_module()
class TPVFormer(Base3DSegmentor):

    def __init__(self,
                 pc_range=None,
                 pc_range_h=None,
                 tpv_h=None,
                 tpv_w=None,
                 tpv_z=None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 backbone=None,
                 neck=None,
                 encoder=None,
                 decode_head=None):

        super().__init__(data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.encoder = MODELS.build(encoder)
        self.pc_range = pc_range
        self.pc_range = pc_range_h
        encoder_high = copy.deepcopy(encoder)
        encoder_high.pc_range = pc_range_h
        # encoder_high.pc_range = [-25.6, -25.6, -2.5, 25.6, 25.6, 1.5]
        # encoder_high.pc_range = [-15, -15, -2.5, 15, 15, 1.5]
        # encoder_high.pc_range = [-18, -18, -2.5, 18, 18, 1.5]
        encoder_high.tpv_h = tpv_h
        encoder_high.tpv_w = tpv_w
        encoder_high.tpv_z = tpv_z
        encoder_high.num_points_in_pillar = [4, 32, 32]
        encoder_high.num_points_in_pillar_cross_view = [16, 16, 16]
        self.encoder_high = MODELS.build(encoder_high)

        self.miu = 0.25
        self.decode_head = MODELS.build(decode_head)

    def extract_feat(self, img):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img_feats = self.backbone(img)

        if hasattr(self, 'neck'):
            img_feats = self.neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped

    def _forward(self, batch_inputs, batch_data_samples):
        """Forward training function."""
        img_feats = self.extract_feat(batch_inputs['imgs'])

        outs_l = self.encoder(img_feats, batch_data_samples)
        outs_h = self.encoder_high(img_feats, batch_data_samples)
        outs_l = self.decode_head(outs_l, batch_inputs['voxels']['coors'])
        outs_h = self.decode_head.forward_h(outs_h, batch_inputs['voxels']['coors'])


        return outs_l, outs_h

    def pre_sampling(self, pc_range, pc_range_h):
        h_range = pc_range[3] - pc_range[0]
        w_range = pc_range[4] - pc_range[1]
        z_range = pc_range[5] - pc_range[2]
        voxel_size_h = h_range / self.encoder.tpv_h
        voxel_size_w = w_range / self.encoder.tpv_w
        voxel_size_z = z_range / self.encoder.tpv_z
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



    def Xcy(self, tpv_list):
        start_h, end_h, start_w, end_w, start_z, end_z = self.pre_sampling(self.encoder.pc_range, self.encoder_high.pc_range)
        len_h = end_h - start_h
        len_w = end_w - start_w
        len_z = end_z - start_z
        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]
        h, w, z = self.encoder_high.tpv_h, self.encoder_high.tpv_w, self.encoder_high.tpv_z
        e_dim = tpv_hw.shape[2]
        tpv_hw = tpv_hw.reshape(1, h, w, e_dim).permute(0, 3, 1, 2)
        tpv_zh = tpv_zh.reshape(1, z, h, e_dim).permute(0, 3, 1, 2)
        tpv_wz = tpv_wz.reshape(1, w, z, e_dim).permute(0, 3, 1, 2)

        tpv_hw = F.adaptive_avg_pool2d(tpv_hw, output_size=(len_h, len_w))
        tpv_zh = F.adaptive_avg_pool2d(tpv_zh, output_size=(len_z, len_h))
        tpv_wz = F.adaptive_avg_pool2d(tpv_wz, output_size=(len_w, len_z))
        # tpv_hw = F.avg_pool2d(tpv_hw, kernel_size=2, stride=2)
        # tpv_zh = F.avg_pool2d(tpv_zh, kernel_size=2, stride=2)
        # tpv_wz = F.avg_pool2d(tpv_wz, kernel_size=2, stride=2)

        tpv_hw = tpv_hw.reshape(1, e_dim, -1).permute(0, 2, 1)
        tpv_zh = tpv_zh.reshape(1, e_dim, -1).permute(0, 2, 1)
        tpv_wz = tpv_wz.reshape(1, e_dim, -1).permute(0, 2, 1)
        tpv_query = [tpv_hw, tpv_zh, tpv_wz]

        return tpv_query

    def loss(self, batch_inputs: dict,
             batch_data_samples: SampleList) -> SampleList:
        img_feats = self.extract_feat(batch_inputs['imgs'])

        # 高分辨率的查询
        queries_high_resolution = self.encoder_high(img_feats, batch_data_samples, mode='high')
        tpv_xcy = self.Xcy(queries_high_resolution)
        # 低分辨率的查询
        queries = self.encoder(img_feats, batch_data_samples, mode='low', tpv_xcy=tpv_xcy, pc_range_h=self.encoder_high.pc_range)

        # 低分辨率的损失
        losses_l = self.decode_head.loss(queries, batch_data_samples)
        # 高分辨率的损失
        losses_h = self.decode_head.loss_h(queries_high_resolution, queries, batch_data_samples, self.encoder_high.pc_range, self.miu)

        for k in losses_l:
            losses_l[k] = 0.2 * losses_l[k]

        for k in losses_h:
            losses_h[k] = 0.8 * losses_h[k]

        losses = {**losses_l, **losses_h}

        return losses

    def predict(self, batch_inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Forward predict function."""
        img_feats = self.extract_feat(batch_inputs['imgs'])
        tpv_queries_high_resolution = self.encoder_high(img_feats, batch_data_samples, mode='high')
        tpv_xcy = self.Xcy(tpv_queries_high_resolution)
        # 低分辨率的查询
        tpv_queries = self.encoder(img_feats, batch_data_samples, mode='low', tpv_xcy=tpv_xcy, pc_range_h=self.encoder_high.pc_range)


        seg_logits_list = self.decode_head.predict(tpv_queries, batch_data_samples)

        seg_logits_list_high = self.decode_head.predict_h(tpv_queries_high_resolution, tpv_queries, batch_data_samples, self.miu)
        # seg_preds = [seg_logit.argmax(dim=1) for seg_logit in seg_logits]

        logits_results = []
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)
            seg_logits_list_high[i] = seg_logits_list_high[i].transpose(0, 1)
            len_high = seg_logits_list_high[i].shape[1]
            logits_result = torch.cat((seg_logits_list_high[i], seg_logits_list[i][:, len_high:]), dim=1)
            logits_results.append(logits_result)
        return self.postprocess_result(logits_results, batch_data_samples)

    def aug_test(self, batch_inputs, batch_data_samples):
        pass

    def encode_decode(self, batch_inputs: dict,
                      batch_data_samples: SampleList) -> SampleList:
        pass

