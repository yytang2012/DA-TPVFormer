from typing import Dict, List, Optional
import copy
import torch
from torch import Tensor

from mmdet3d.models import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d.utils import get_lidar2img
from projects.BEVFormer.bevformer.grid_mask import GridMask


@MODELS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
        Args:
            video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 rescale=False,
                 video_test_mode=False,
                 data_preprocessor=None,
                 init_cfg=None,
                 **kwargs
                 ):

        super().__init__(
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_fusion_layer=pts_fusion_layer,
            img_backbone=img_backbone,
            pts_backbone=pts_backbone,
            img_neck=img_neck,
            pts_neck=pts_neck,
            pts_bbox_head=pts_bbox_head,
            img_roi_head=img_roi_head,
            img_rpn_head=img_rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs
        )
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.rescale = rescale

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def extract_img_feat(self, img: Tensor,
                         batch_input_metas: List[dict] = None,
                         len_queue=None) -> List[Tensor]:
        """Extract features from images.

        Args:
            len_queue:
            img (tensor): Batched multi-view image tensor with
                shape (B, N, C, H, W).
            batch_input_metas (list[dict]): Meta information.

        Returns:
            list[tensor]: Multi-level image features.
        """
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    def extract_feat(self,
                     batch_inputs_dict: Dict[str, Optional[Tensor]],
                     batch_input_metas: List[dict]) -> List[Tensor]:
        """Extract features from images.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `imgs` keys.
            batch_input_metas (list[dict]): Meta information of multiple inputs.

        Returns:
            list[tensor]: Multi-level image features.
        """
        imgs = batch_inputs_dict.get('imgs', None)
        img_feats = self.extract_img_feat(imgs, batch_input_metas)
        return img_feats

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def obtain_history_bev(self, prev_samples):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()
        image_list = [_.get('inputs').get('imgs') for _ in prev_samples]
        meta_list = [_.get('data_samples')[0].metainfo for _ in prev_samples]
        self.add_lidar2img(meta_list)

        # [bs, N, C, H, W] -> [bs, 1, N, C, H, W]
        expanded_images = [img.unsqueeze(1) for img in image_list]

        # The resulting shape becomes [bs, Len_queue, N, C, H, W]
        images = torch.cat(expanded_images, dim=1)

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = images.shape
            images = images.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_img_feat(img=images, len_queue=len_queue)
            for i in range(len_queue):
                _meta = meta_list[i]
                if not _meta['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, [_meta], prev_bev, only_bev=True)
            self.train()
            return prev_bev

    def forward(self, *args, mode: str = 'tensor', **kwargs):
        if mode == 'loss':
            return self.loss(args, **kwargs)
        elif mode == 'predict':
            return self.predict(**kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". Only supports loss and predict mode')

    def loss(self, data_list, **kwargs):
        # Process the current sample
        current_sample = data_list[-1]
        inputs = current_sample.get("inputs")
        data_samples = current_sample.get("data_samples")
        input_metas = [item.metainfo for item in data_samples]
        input_metas = self.add_lidar2img(input_metas)
        if not input_metas[0]['prev_bev_exists']:
            prev_bev = None
        else:
            prev_samples = data_list[:-1]
            prev_bev = self.obtain_history_bev(prev_samples)

        img_feats = self.extract_feat(batch_inputs_dict=inputs, batch_input_metas=input_metas)

        outs = self.pts_bbox_head(img_feats, input_metas, prev_bev)

        batch_gt_instances_3d = [
            item.gt_instances_3d for item in data_samples
        ]
        loss_inputs = [batch_gt_instances_3d, outs]
        losses_pts = self.pts_bbox_head.loss_by_feat(*loss_inputs)

        return losses_pts

    def predict(self, inputs=None, data_samples=None, **kwargs):
        """Forward of testing.

        Args:
            inputs (dict): The model input dict which include
                `imgs` keys.
            data_samples (List[:obj:`Det3DDataSample`]): The Data Samples.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the input sample.
        """

        # Get meta information
        input_metas = [item.metainfo for item in data_samples]
        input_metas = self.add_lidar2img(input_metas)
        input_metas = self.add_motion_info(input_metas)  # Add motion and temporal information

        img_feats = self.extract_feat(batch_inputs_dict=inputs, batch_input_metas=input_metas)

        outs = self.pts_bbox_head(img_feats, input_metas, prev_bev=self.prev_frame_info['prev_bev'])

        results_list_3d = self.pts_bbox_head.predict_by_feat(outs, input_metas, rescale=self.rescale)

        self.prev_frame_info['prev_bev'] = outs['bev_embed'] if self.video_test_mode else None

        data_samples = self.add_pred_to_datasample(data_samples, results_list_3d)

        return data_samples

    def add_motion_info(self, batch_input_metas: List[Dict]) -> List[Dict]:
        """Add temporal and ego motion information into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs.

        Returns:
            batch_input_metas (list[dict]): Meta info with temporal and motion info added.
        """
        # Handle scene switching
        if batch_input_metas[0]['scene_token'] != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_bev'] = None

        # Update scene token
        self.prev_frame_info['scene_token'] = batch_input_metas[0]['scene_token']

        # Store current ego motion
        tmp_pos = copy.deepcopy(batch_input_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(batch_input_metas[0]['can_bus'][-1])

        if self.prev_frame_info['prev_bev'] is not None:
            # Calculate relative motion
            batch_input_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            batch_input_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            # Reset motion for first frame
            batch_input_metas[0]['can_bus'][-1] = 0
            batch_input_metas[0]['can_bus'][:3] = 0

        # Update history position and angle
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle

        # # Add prev_bev to meta info
        for meta in batch_input_metas:
            meta['prev_bev'] = self.prev_frame_info['prev_bev'] if self.video_test_mode else None

        return batch_input_metas

    # may need speed-up
    def add_lidar2img(self, batch_input_metas: List[Dict]) -> List[Dict]:
        """add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        for meta in batch_input_metas:
            l2i = list()
            for i in range(len(meta['cam2img'])):
                c2i = torch.tensor(meta['cam2img'][i]).double()
                l2c = torch.tensor(meta['lidar2cam'][i]).double()
                l2i.append(get_lidar2img(c2i, l2c).float().numpy())
            meta['lidar2img'] = l2i
        return batch_input_metas
