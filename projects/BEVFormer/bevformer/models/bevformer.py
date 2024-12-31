from typing import Dict, List, Optional, Union
import copy
import torch
from mmengine.structures import InstanceData
from torch import Tensor
from mmdet3d.structures.det3d_data_sample import (ForwardResults,
                                                  OptSampleList, SampleList)

from mmdet3d.models import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample, bbox3d2result
from mmdet3d.structures.bbox_3d.utils import get_lidar2img
from projects.BEVFormer.bevformer.utils.grid_mask import GridMask


@MODELS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
        Args:
            video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
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
                 pretrained=None,
                 video_test_mode=False,
                 data_preprocessor=None,  # 添加 data_preprocessor
                 init_cfg=None,  # 添加 init_cfg
                 **kwargs
                 ):

        # super().__init__(pts_voxel_layer, pts_voxel_encoder,
        #                  pts_middle_encoder, pts_fusion_layer,
        #                  img_backbone, pts_backbone, img_neck, pts_neck,
        #                  pts_bbox_head, img_roi_head, img_rpn_head,
        #                  train_cfg, test_cfg, pretrained)
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
            init_cfg=init_cfg,  # 添加 init_cfg
            data_preprocessor=data_preprocessor,  # 添加 data_preprocessor
            **kwargs
        )
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    # def extract_img_feat(self, img, img_metas, len_queue=None):
    #     """Extract features of images."""
    #     B = img.size(0)
    #     if img is not None:
    #
    #         # input_shape = img.shape[-2:]
    #         # # update real input shape of each single img
    #         # for img_meta in img_metas:
    #         #     img_meta.update(input_shape=input_shape)
    #
    #         if img.dim() == 5 and img.size(0) == 1:
    #             img.squeeze_()
    #         elif img.dim() == 5 and img.size(0) > 1:
    #             B, N, C, H, W = img.size()
    #             img = img.reshape(B * N, C, H, W)
    #         if self.use_grid_mask:
    #             img = self.grid_mask(img)
    #
    #         img_feats = self.img_backbone(img)
    #         if isinstance(img_feats, dict):
    #             img_feats = list(img_feats.values())
    #     else:
    #         return None
    #     if self.with_img_neck:
    #         img_feats = self.img_neck(img_feats)
    #
    #     img_feats_reshaped = []
    #     for img_feat in img_feats:
    #         BN, C, H, W = img_feat.size()
    #         if len_queue is not None:
    #             img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
    #         else:
    #             img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
    #     return img_feats_reshaped

    def extract_img_feat(self, img: Tensor,
                         batch_input_metas: List[dict]) -> List[Tensor]:
        """Extract features from images.

        Args:
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
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    # def extract_feat(self, img, img_metas=None, len_queue=None):
    #     """Extract features from images and points."""
    #
    #     img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
    #
    #     return img_feats

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
        img = batch_inputs_dict['img']
        if isinstance(img, list):
            img = torch.stack(img, dim=0)
        img_feats = self.extract_img_feat(img, batch_input_metas)
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    # def forward(self,
    #             inputs: Union[dict, List[dict]],
    #             data_samples: OptSampleList = None,
    #             mode: str = 'tensor',
    #             **kwargs) -> ForwardResults:
    #     if mode == 'loss':
    #         return self.loss(inputs, data_samples, **kwargs)
    #     elif mode == 'predict':
    #         return self.predict(inputs, data_samples, **kwargs)
    #     elif mode == 'tensor':
    #         return self._forward(inputs, data_samples, **kwargs)
    #     else:
    #         raise RuntimeError(f'Invalid mode "{mode}". '
    #                            'Only supports loss, predict and tensor mode')

    # def forward(self,
    #             data,
    #             mode: str = 'predict',
    #             **kwargs) -> ForwardResults:
    #     if mode == 'loss':
    #         return self.loss(**data)
    #     elif mode == 'predict':
    #         return self.predict(**data)
    #     # elif mode == 'tensor':
    #     #     return self._forward(inputs, data_samples, **kwargs)
    #     else:
    #         raise RuntimeError(f'Invalid mode "{mode}". '
    #                            'Only supports loss, predict and tensor mode')

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    def loss(self,
             points=None,
             img_metas=None,
             gt_bboxes_3d=None,
             gt_labels_3d=None,
             gt_labels=None,
             gt_bboxes=None,
             img=None,
             proposals=None,
             gt_bboxes_ignore=None,
             img_depth=None,
             img_mask=None,
             ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    # def predict(self, img_metas, img=None, **kwargs):
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
        batch_input_metas = [item.metainfo for item in data_samples]
        batch_input_metas = self.add_lidar2img(batch_input_metas)

        # Add motion and temporal information
        batch_input_metas = self.add_motion_info(batch_input_metas)

        new_prev_bev, results_list_3d = self.simple_test(
            batch_input_metas=batch_input_metas, batch_inputs_dict=inputs, prev_bev=self.prev_frame_info['prev_bev'],
            **kwargs)
        self.prev_frame_info['prev_bev'] = new_prev_bev

        for i, data_sample in enumerate(data_samples):
            results_list_3d_i = InstanceData(
                metainfo=results_list_3d[i]['pts_bbox'])
            data_sample.pred_instances_3d = results_list_3d_i
            data_sample.pred_instances = InstanceData()

        return data_samples

        # # Extract features
        # img_feats = self.extract_feat(inputs, batch_input_metas)
        #
        # # Get model outputs
        # outs = self.pts_bbox_head(img_feats, batch_input_metas,
        #                           prev_bev=batch_input_metas[0]['prev_bev'])
        #
        # # Store BEV features for next frame
        # self.prev_frame_info['prev_bev'] = outs['bev_embed']
        #
        # # Get 3D detection results
        # results_list_3d = self.pts_bbox_head.predict_by_feat(
        #     outs, batch_input_metas, **kwargs)
        #
        # # Format results
        # det_samples = self.add_pred_to_datasample(data_samples,
        #                                           results_list_3d)

        # return det_samples
        #
        # # TODO: fix the following
        # for var, name in [(img_metas, 'img_metas')]:
        #     if not isinstance(var, list):
        #         raise TypeError('{} must be a list, but got {}'.format(
        #             name, type(var)))
        # img = [img] if img is None else img
        #
        # if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
        #     # the first sample of each scene is truncated
        #     self.prev_frame_info['prev_bev'] = None
        # # update idx
        # self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']
        #
        # # do not use temporal information
        # if not self.video_test_mode:
        #     self.prev_frame_info['prev_bev'] = None
        #
        # # Get the delta of ego position and angle between two timestamps.
        # tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        # tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        # if self.prev_frame_info['prev_bev'] is not None:
        #     img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
        #     img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        # else:
        #     img_metas[0][0]['can_bus'][-1] = 0
        #     img_metas[0][0]['can_bus'][:3] = 0
        #
        # new_prev_bev, bbox_results = self.simple_test(
        #     img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # # During inference, we save the BEV features and ego motion of each timestamp.
        # self.prev_frame_info['prev_pos'] = tmp_pos
        # self.prev_frame_info['prev_angle'] = tmp_angle
        # self.prev_frame_info['prev_bev'] = new_prev_bev
        # return bbox_results

    def simple_test_pts(self, x, batch_input_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, batch_input_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, batch_input_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    #
    def simple_test(self, batch_inputs_dict, batch_input_metas=None, prev_bev=None, rescale=True):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(batch_inputs_dict=batch_inputs_dict, batch_input_metas=batch_input_metas)

        bbox_list = [dict() for i in range(len(batch_input_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, batch_input_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list

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
        # for meta in batch_input_metas:
        #     meta['prev_bev'] = self.prev_frame_info['prev_bev'] if self.video_test_mode else None

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
