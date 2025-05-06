from typing import Optional, Union

from torch import nn

from mmdet3d.models import Base3DSegmentor
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
import copy
import torch
import torch.nn.functional as F
@MODELS.register_module()
class TPVFormer(Base3DSegmentor):

    def __init__(self,
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
        encoder_high = copy.deepcopy(encoder)
        encoder_high.pc_range = [-25.6, -25.6, -2.5, 25.6, 25.6, 1.5]
        # encoder_high.pc_range = [-15, -15, -2.5, 15, 15, 1.5]
        # encoder_high.pc_range = [-18, -18, -2.5, 18, 18, 1.5]
        encoder_high.tpv_h = 100
        encoder_high.tpv_w = 100
        encoder_high.tpv_z = 8
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

    def Xcy(self, tpv_list):
        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]
        h, w, z = self.encoder_high.tpv_h, self.encoder_high.tpv_w, self.encoder_high.tpv_z
        e_dim = tpv_hw.shape[2]
        tpv_hw = tpv_hw.reshape(1, h, w, e_dim).permute(0, 3, 1, 2)
        tpv_zh = tpv_zh.reshape(1, z, h, e_dim).permute(0, 3, 1, 2)
        tpv_wz = tpv_wz.reshape(1, w, z, e_dim).permute(0, 3, 1, 2)

        tpv_hw = F.avg_pool2d(tpv_hw, kernel_size=2, stride=2)
        tpv_zh = F.avg_pool2d(tpv_zh, kernel_size=2, stride=2)
        tpv_wz = F.avg_pool2d(tpv_wz, kernel_size=2, stride=2)

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
        queries = self.encoder(img_feats, batch_data_samples, mode='low', tpv_xcy=tpv_xcy)

        # 低分辨率的损失
        losses = self.decode_head.loss(queries, queries_high_resolution, batch_data_samples, self.miu)

        # losses = {**losses_l, **losses_h}

        return losses

    def predict(self, batch_inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Forward predict function."""
        img_feats = self.extract_feat(batch_inputs['imgs'])
        tpv_queries_high_resolution = self.encoder_high(img_feats, batch_data_samples, mode='high')
        tpv_xcy = self.Xcy(tpv_queries_high_resolution)
        # 低分辨率的查询
        tpv_queries = self.encoder(img_feats, batch_data_samples, mode='low', tpv_xcy=tpv_xcy)

        seg_logits_list = self.decode_head.predict(tpv_queries, tpv_queries_high_resolution, batch_data_samples, self.miu)

        # seg_logits_list_high = self.decode_head.predict_h(tpv_queries_high_resolution, batch_data_samples)
        # seg_preds = [seg_logit.argmax(dim=1) for seg_logit in seg_logits]

        logits_results = []
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)

        return self.postprocess_result(seg_logits_list, batch_data_samples)

    def aug_test(self, batch_inputs, batch_data_samples):
        pass

    def encode_decode(self, batch_inputs: dict,
                      batch_data_samples: SampleList) -> SampleList:
        pass

