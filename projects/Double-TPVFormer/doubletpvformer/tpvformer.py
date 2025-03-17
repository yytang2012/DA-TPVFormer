from typing import Optional, Union

from torch import nn

from mmdet3d.models import Base3DSegmentor
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
import copy
import torch

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

        self.encoder_high = MODELS.build(encoder)
        self.encoder_high.pc_range = [-25.6, -25.6, -2.5, 25.6, 25.6, 1.5]
        self.encoder_high.tpv_h = 100
        self.encoder_high.tpv_w = 100
        self.encoder_high.tpv_z = 16
        self.encoder_high.num_points_in_pillar = [4, 32, 32]
        self.encoder_high.num_points_in_pillar_cross_view = [16, 16, 16]

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
        img_feats_low = [feat[:, :6, ...] for feat in img_feats]  # 原始6张图的特征
        img_feats_high = [feat[:, 6:, ...] for feat in img_feats]  # 裁剪放大6张图的特征
        outs_l = self.encoder(img_feats_low, batch_data_samples)
        outs_h = self.encoder_high(img_feats_high, batch_data_samples)
        outs_l = self.decode_head(outs_l, batch_inputs['voxels']['coors'])
        outs_h = self.decode_head.forward_h(outs_h, batch_inputs['voxels']['coors'])


        return outs_l, outs_h

    def loss(self, batch_inputs: dict,
             batch_data_samples: SampleList) -> SampleList:
        img_feats = self.extract_feat(batch_inputs['imgs'])
        # 拆分每个尺度的特征图
        img_feats_low = [feat[:, :6, ...] for feat in img_feats]  # 原始6张图的特征
        img_feats_high = [feat[:, 6:, ...] for feat in img_feats]  # 裁剪放大6张图的特征
        # 低分辨率的查询
        queries = self.encoder(img_feats_low, batch_data_samples)
        # 高分辨率的查询
        queries_high_resolution = self.encoder_high(img_feats_high, batch_data_samples)
        # 低分辨率的损失
        losses_l = self.decode_head.loss(queries, batch_data_samples)
        # 高分辨率的损失
        losses_h = self.decode_head.loss_h(queries_high_resolution, batch_data_samples)
        losses = {**losses_l, **losses_h}

        return losses

    def predict(self, batch_inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Forward predict function."""
        img_feats = self.extract_feat(batch_inputs['imgs'])

        img_feats_low = [feat[:, :6, ...] for feat in img_feats]  # 原始6张图的特征
        img_feats_high = [feat[:, 6:, ...] for feat in img_feats]  # 裁剪放大6张图的特征

        tpv_queries = self.encoder(img_feats_low, batch_data_samples)
        tpv_queries_high_resolution = self.encoder_high(img_feats_high, batch_data_samples)

        seg_logits_list = self.decode_head.predict(tpv_queries, batch_data_samples)

        seg_logits_list_high = self.decode_head.predict_h(tpv_queries_high_resolution, batch_data_samples)
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
