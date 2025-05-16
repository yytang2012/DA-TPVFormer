# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Det3DDataPreprocessor
from mmdet3d.models.data_preprocessors.voxelize import dynamic_scatter_3d
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape
from mmdet3d.utils import OptConfigType
from mmdet.models.utils.misc import samplelist_boxtype2tensor

@MODELS.register_module()
class TPVFormerDataPreprocessor(Det3DDataPreprocessor):

    def __init__(self,
                 voxel_h: bool = False,
                 voxel_layer_h: OptConfigType = None,
                 tpv_w: int = 200,
                 tpv_h: int = 200,
                 tpv_z: int = 16,
                 fill_labels: int = 0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.voxel_h = voxel_h
        self.tpv_w = tpv_w
        self.tpv_h = tpv_h
        self.tpv_z = tpv_z
        self.fill_labels = fill_labels

        if voxel_h:
            self.voxel_layer_h = VoxelizationByGridShape(**voxel_layer_h)

    @torch.no_grad()
    def simple_process(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion for img data
        based on ``BaseDataPreprocessor``, and voxelize point cloud if `voxel`
        is set to be True.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)

        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

            if self.voxel:
                voxel_dict = self.voxelize(inputs['points'], data_samples)
                batch_inputs['voxels'] = voxel_dict

        if 'points_h' in inputs:
            batch_inputs['points_h'] = inputs['points_h']

            if self.voxel_h:
                voxel_dict = self.voxelize_h(inputs['points_h'], data_samples)
                batch_inputs['voxels_h'] = voxel_dict

        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                # in DETR, this is needed for the construction of masks, which
                # is then used for the transformer_head.
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if self.boxtype2tensor:
                    samplelist_boxtype2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)
                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs['imgs'] = imgs

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    @torch.no_grad()
    def voxelize(self, points: List[Tensor],
                 data_samples: SampleList) -> List[Tensor]:
        """Apply voxelization to point cloud. In TPVFormer, it will get voxel-
        wise segmentation label and voxel/point coordinates.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (List[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            List[Tensor]: Coordinates of voxels, shape is Nx3,
        """

        for point, data_sample in zip(points, data_samples):
            min_bound = point.new_tensor(
                self.voxel_layer.point_cloud_range[:3])

            max_bound = point.new_tensor(
                self.voxel_layer.point_cloud_range[3:])

            point_clamp = torch.clamp(point, min_bound, max_bound + 1e-6)

            coors = torch.floor(
                (point_clamp - min_bound) /
                point_clamp.new_tensor(self.voxel_layer.voxel_size)).int()

            self.get_voxel_seg(coors, data_sample)

            data_sample.point_coors = coors


    @torch.no_grad()
    def voxelize_h(self, points_h: List[Tensor],
                 data_samples: SampleList) -> List[Tensor]:
        """Apply voxelization to point cloud. In TPVFormer, it will get voxel-
        wise segmentation label and voxel/point coordinates.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (List[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            List[Tensor]: Coordinates of voxels, shape is Nx3,
        """
        for point_h, data_sample in zip(points_h, data_samples):

            min_bound_h = point_h.new_tensor(
                self.voxel_layer_h.point_cloud_range[:3])

            max_bound_h = point_h.new_tensor(
                self.voxel_layer_h.point_cloud_range[3:])

            point_clamp_h = torch.clamp(point_h, min_bound_h, max_bound_h + 1e-6)

            coors_h = torch.floor(
                (point_clamp_h - min_bound_h) /
                point_clamp_h.new_tensor(self.voxel_layer_h.voxel_size)).int()
            self.get_voxel_seg_h(coors_h, data_sample)


            data_sample.point_coors_h = coors_h

    def get_voxel_seg(self, res_coors: Tensor, data_sample: SampleList):
        """Get voxel-wise segmentation label and point2voxel map.

        Args:
            res_coors (Tensor): The voxel coordinates of points, Nx3.
            data_sample: (:obj:`Det3DDataSample`): The annotation data of
                every samples. Add voxel-wise annotation forsegmentation.
        """

        if self.training:
            pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
            pts_semantic_mask = F.one_hot(pts_semantic_mask.long()).float()
            voxel_semantic_mask, voxel_coors, point2voxel_map = \
                dynamic_scatter_3d(pts_semantic_mask, res_coors, 'mean', True)
            voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)

            grid_shape = (self.tpv_w, self.tpv_h, self.tpv_z)
            fill_labels = self.fill_labels

            # 初始化稠密体素语义标签 grid（全设为 empty 类）
            dense_voxel_sem_mask = torch.full(grid_shape, fill_value=fill_labels, dtype=torch.long,
                                              device=voxel_coors.device)

            # voxel_coors 的 shape 是 (M, 3)，表示 M 个非空体素的坐标 (x, y, z)
            # voxel_semantic_mask 的 shape 是 (M,)，表示对应的语义类别
            x, y, z = voxel_coors[:, 0], voxel_coors[:, 1], voxel_coors[:, 2]
            dense_voxel_sem_mask[x, y, z] = voxel_semantic_mask
            # num_nonzero = (dense_voxel_sem_mask != 0).sum()

            dense_voxel_sem_mask = dense_voxel_sem_mask.reshape(-1)

            # 生成每个维度的坐标轴
            x_range = torch.arange(grid_shape[0], device=dense_voxel_sem_mask.device)
            y_range = torch.arange(grid_shape[1], device=dense_voxel_sem_mask.device)
            z_range = torch.arange(grid_shape[2], device=dense_voxel_sem_mask.device)
            # 网格坐标生成，注意使用 indexing='ij' 保持 XYZ 对应
            xx, yy, zz = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
            # 合并为 (N, 3) 的体素坐标列表
            dense_voxel_coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # shape: (100*100*8, 3)

            data_sample.gt_pts_seg.voxel_semantic_mask = dense_voxel_sem_mask
            data_sample.point2voxel_map = point2voxel_map
            data_sample.voxel_coors = dense_voxel_coords
        else:
            pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
            pts_semantic_mask = F.one_hot(pts_semantic_mask.long()).float()
            voxel_semantic_mask, voxel_coors, point2voxel_map = \
                dynamic_scatter_3d(pts_semantic_mask, res_coors, 'mean', True)
            voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
            data_sample.gt_pts_seg.voxel_semantic_mask = voxel_semantic_mask
            data_sample.point2voxel_map = point2voxel_map
            data_sample.voxel_coors = voxel_coors

        # else:
        #     pseudo_tensor = res_coors.new_ones([res_coors.shape[0], 1]).float()
        #     _, _, point2voxel_map = dynamic_scatter_3d(pseudo_tensor,
        #                                                res_coors, 'mean', True)
        #     data_sample.point2voxel_map = point2voxel_map

    def get_voxel_seg_h(self, res_coors: Tensor, data_sample: SampleList):
        """Get voxel-wise segmentation label and point2voxel map.

        Args:
            res_coors (Tensor): The voxel coordinates of points, Nx3.
            data_sample: (:obj:`Det3DDataSample`): The annotation data of
                every samples. Add voxel-wise annotation forsegmentation.
        """

        if self.training:
            pts_semantic_mask_h = data_sample.gt_pts_seg.pts_semantic_mask_h
            pts_semantic_mask_h = F.one_hot(pts_semantic_mask_h.long()).float()
            voxel_semantic_mask_h, voxel_coors_h, point2voxel_map_h = \
                dynamic_scatter_3d(pts_semantic_mask_h, res_coors, 'mean', True)
            voxel_semantic_mask_h = torch.argmax(voxel_semantic_mask_h, dim=-1)

            grid_shape = (self.tpv_w, self.tpv_h, self.tpv_z)
            fill_labels = self.fill_labels

            # 初始化稠密体素语义标签 grid（全设为 empty 类）
            dense_voxel_sem_mask_h = torch.full(grid_shape, fill_value=fill_labels, dtype=torch.long,
                                              device=voxel_coors_h.device)

            # voxel_coors 的 shape 是 (M, 3)，表示 M 个非空体素的坐标 (x, y, z)
            # voxel_semantic_mask 的 shape 是 (M,)，表示对应的语义类别
            x, y, z = voxel_coors_h[:, 0], voxel_coors_h[:, 1], voxel_coors_h[:, 2]
            dense_voxel_sem_mask_h[x, y, z] = voxel_semantic_mask_h
            # num_nonzero = (dense_voxel_sem_mask != 0).sum()

            dense_voxel_sem_mask_h = dense_voxel_sem_mask_h.reshape(-1)

            # 生成每个维度的坐标轴
            x_range = torch.arange(grid_shape[0], device=dense_voxel_sem_mask_h.device)
            y_range = torch.arange(grid_shape[1], device=dense_voxel_sem_mask_h.device)
            z_range = torch.arange(grid_shape[2], device=dense_voxel_sem_mask_h.device)
            # 网格坐标生成，注意使用 indexing='ij' 保持 XYZ 对应
            xx, yy, zz = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
            # 合并为 (N, 3) 的体素坐标列表
            dense_voxel_coords_h = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # shape: (100*100*8, 3)

            data_sample.gt_pts_seg.voxel_semantic_mask_h = dense_voxel_sem_mask_h
            data_sample.point2voxel_map_h = point2voxel_map_h
            data_sample.voxel_coors_h = dense_voxel_coords_h
        else:
            pts_semantic_mask_h = data_sample.gt_pts_seg.pts_semantic_mask_h
            pts_semantic_mask_h = F.one_hot(pts_semantic_mask_h.long()).float()
            voxel_semantic_mask_h, voxel_coors_h, point2voxel_map_h = \
                dynamic_scatter_3d(pts_semantic_mask_h, res_coors, 'mean', True)
            voxel_semantic_mask_h = torch.argmax(voxel_semantic_mask_h, dim=-1)
            data_sample.gt_pts_seg.voxel_semantic_mask_h = voxel_semantic_mask_h
            data_sample.point2voxel_map_h = point2voxel_map_h
            data_sample.voxel_coors_h = voxel_coors_h

        # else:
        #     pseudo_tensor = res_coors.new_ones([res_coors.shape[0], 1]).float()
        #     _, _, point2voxel_map_h = dynamic_scatter_3d(pseudo_tensor,
        #                                                res_coors, 'mean', True)
        #     data_sample.point2voxel_map_h = point2voxel_map_h

@MODELS.register_module()
class GridMask(nn.Module):
    """GridMask data augmentation.

        Modified from https://github.com/dvlab-research/GridMask.

    Args:
        use_h (bool): Whether to mask on height dimension. Defaults to True.
        use_w (bool): Whether to mask on width dimension. Defaults to True.
        rotate (int): Rotation degree. Defaults to 1.
        offset (bool): Whether to mask offset. Defaults to False.
        ratio (float): Mask ratio. Defaults to 0.5.
        mode (int): Mask mode. if mode == 0, mask with square grid.
            if mode == 1, mask the rest. Defaults to 0
        prob (float): Probability of applying the augmentation.
            Defaults to 1.0.
    """

    def __init__(self,
                 use_h: bool = True,
                 use_w: bool = True,
                 rotate: int = 1,
                 offset: bool = False,
                 ratio: float = 0.5,
                 mode: int = 0,
                 prob: float = 1.0):
        super().__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def forward(self, inputs: Tensor,
                data_samples: SampleList) -> Tuple[Tensor, SampleList]:
        if np.random.rand() > self.prob:
            return inputs, data_samples
        height, width = inputs.shape[-2:]
        mask_height = int(1.5 * height)
        mask_width = int(1.5 * width)
        distance = np.random.randint(2, min(height, width))
        length = min(max(int(distance * self.ratio + 0.5), 1), distance - 1)
        mask = np.ones((mask_height, mask_width), np.float32)
        stride_on_height = np.random.randint(distance)
        stride_on_width = np.random.randint(distance)
        if self.use_h:
            for i in range(mask_height // distance):
                start = distance * i + stride_on_height
                end = min(start + length, mask_height)
                mask[start:end, :] *= 0
        if self.use_w:
            for i in range(mask_width // distance):
                start = distance * i + stride_on_width
                end = min(start + length, mask_width)
                mask[:, start:end] *= 0

        # NOTE: r is the rotation radian, here is a random counterclockwise
        # rotation of 1° or remain unchanged, which follows the implementation
        # of the official detection version.
        # https://github.com/dvlab-research/GridMask.
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))

        mask = mask.rotate(r)
        mask = np.array(mask)
        mask = mask[int(0.25 * height):int(0.25 * height) + height,
                    int(0.25 * width):int(0.25 * width) + width]

        mask = inputs.new_tensor(mask)
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(inputs)
        if self.offset:
            offset = inputs.new_tensor(2 *
                                       (np.random.rand(height, width) - 0.5))
            inputs = inputs * mask + offset * (1 - mask)
        else:
            inputs = inputs * mask

        return inputs, data_samples
