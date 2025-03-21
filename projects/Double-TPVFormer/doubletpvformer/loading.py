# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Union

import mmcv
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmengine.fileio import get

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles, Pack3DDetInputs
from mmdet3d.registry import TRANSFORMS

import cv2
from typing import List, Sequence, Union
import mmengine
from numpy import dtype
from mmdet3d.structures import BaseInstance3DBoxes, Det3DDataSample, PointData
from mmdet3d.structures.points import BasePoints, LiDARPoints
from mmengine.structures import InstanceData
import torch

Number = Union[int, float]

def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        if data.dtype is dtype('float64'):
            data = data.astype(np.float32)
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename, cam2img, lidar2cam, lidar2img = [], [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])

            lidar2cam_array = np.array(cam_item['lidar2cam'])
            cam2img_array = np.eye(4).astype(np.float64)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img'])
            cam2img.append(cam2img_array)
            lidar2img.append(cam2img_array @ lidar2cam_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        # gbr follow tpvformer
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results


@TRANSFORMS.register_module()
class SegLabelMapping(BaseTransform):
    """Map original semantic class to valid category ids.

    Required Keys:

    - seg_label_mapping (np.ndarray)
    - pts_semantic_mask (np.ndarray)

    Added Keys:

    - points (np.float32)

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).
    """

    def transform(self, results: dict) -> dict:
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
            Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        assert 'seg_label_mapping' in results
        label_mapping = results['seg_label_mapping']
        converted_pts_sem_mask = np.vectorize(
            label_mapping.__getitem__, otypes=[np.uint8])(
            pts_semantic_mask)

        results['pts_semantic_mask'] = converted_pts_sem_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            assert 'pts_semantic_mask' in results['eval_ann_info']
            results['eval_ann_info']['pts_semantic_mask'] = \
                converted_pts_sem_mask

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class PointsBoxFilter(BaseTransform):
    """Filter points by a 3D box range.
    Args:
        point_box_type (tuple): A tuple of three tuples, each containing min and max values
            for x, y, and z dimensions respectively. Use None for no limit in a dimension.
            Format: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        keep_inside (bool): If True, keep points inside the box. If False, keep points outside the box.
    """

    def __init__(self, point_box_type=((None, None), (None, None), (None, None)), keep_inside=True):
        self.point_box_type = point_box_type
        self.keep_inside = keep_inside
        super().__init__()

    def transform(self, results: dict) -> dict:
        """Transform function to filter points.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
            and 'pts_semantic_mask' keys are updated in the result dict.
        """

        points = results['points']
        coords = points.coord.numpy()

        # Initialize mask as all True
        mask = np.ones(len(coords), dtype=bool)

        for i, (min_val, max_val) in enumerate(self.point_box_type):
            if min_val is not None:
                mask &= (coords[:, i] >= min_val)
            if max_val is not None:
                mask &= (coords[:, i] < max_val)


        # Invert mask if we want to keep points outside the box
        # if not self.keep_inside:
        mask_l = ~mask
        mask_h = mask

        # Filter points

        results['points_l'] = points[mask_l]
        results['points_h'] = points[mask_h]
        merge_results = np.concatenate(
            (results['points_h'], results['points_l']), axis=0
        )
        results['points'] = LiDARPoints(merge_results)



        # Filter instance and semantic masks if they exist
        if 'pts_instance_mask' in results:
            results['pts_instance_mask_h'] = results['pts_instance_mask'][mask_h]
            results['pts_instance_mask_l'] = results['pts_instance_mask'][mask_l]
            results['pts_instance_mask'] = np.concatenate(
                (results['pts_instance_mask_h'], results['pts_instance_mask_l']), axis=0
            )

        if 'pts_semantic_mask' in results:
            results['pts_semantic_mask_h'] = results['pts_semantic_mask'][mask_h]
            results['pts_semantic_mask_l'] = results['pts_semantic_mask'][mask_l]
            results['pts_semantic_mask'] = np.concatenate(
                (results['pts_semantic_mask_h'], results['pts_semantic_mask_l']), axis=0
            )

            if 'eval_ann_info' in results and 'pts_semantic_mask' in results['eval_ann_info']:
                # results['eval_ann_info']['pts_semantic_mask'] = results['eval_ann_info']['pts_semantic_mask'][mask]
                results['eval_ann_info']['pts_semantic_mask_h'] = results['eval_ann_info']['pts_semantic_mask'][mask_h]
                results['eval_ann_info']['pts_semantic_mask_l'] = results['eval_ann_info']['pts_semantic_mask'][mask_l]
                results['eval_ann_info']['pts_semantic_mask'] = np.concatenate(
                    (results['eval_ann_info']['pts_semantic_mask_h'], results['eval_ann_info']['pts_semantic_mask_l']), axis=0
                )

        # TODO: Uncomment and adjust this part if you want to filter bounding boxes
        # if 'gt_bboxes_3d' in results:
        #     gt_bboxes_3d = results['gt_bboxes_3d']
        #     gt_labels_3d = results['gt_labels_3d']
        #
        #     # Get box centers
        #     centers = gt_bboxes_3d.gravity_center.numpy()
        #
        #     # Initialize box mask as all True
        #     box_mask = np.ones(len(centers), dtype=bool)
        #
        #     # Apply filtering for each dimension
        #     for i, (min_val, max_val) in enumerate(self.point_box_type):
        #         if min_val is not None:
        #             box_mask &= (centers[:, i] >= min_val)
        #         if max_val is not None:
        #             box_mask &= (centers[:, i] < max_val)
        #
        #     # Invert box_mask if we want to keep boxes outside the specified range
        #     if not self.keep_inside:
        #         box_mask = ~box_mask
        #
        #     # Filter bounding boxes and labels
        #     results['gt_bboxes_3d'] = gt_bboxes_3d[box_mask]
        #     results['gt_labels_3d'] = gt_labels_3d[box_mask]

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_box_type={self.point_box_type}, '
        repr_str += f'keep_inside={self.keep_inside})'
        return repr_str

@TRANSFORMS.register_module()
class DuplicateAndCropImages(BaseTransform):
    """Duplicate each input image by cropping the center and resizing."""

    def __init__(self, crop_ratio=0.5):
        """
        Args:
            crop_ratio (float): The ratio of the cropped area compared to the original image.
                                Default is 0.5, meaning the center region is half the width & height.
        """
        self.crop_ratio = crop_ratio

    def transform(self, results):
        """
        Args:
            results (dict): The dictionary containing 'img' as a list of images.

        Returns:
            dict: Updated results with 12 images instead of 6.
        """
        new_images = []

        for img in results['img']:
            # Convert image to NumPy array if needed
            if not isinstance(img, np.ndarray):
                img = np.array(img)

            h, w = img.shape[:2]

            # Compute crop region
            crop_h, crop_w = int(h * self.crop_ratio), int(w * self.crop_ratio)
            start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2

            # Crop the center region
            cropped_img = img[start_h:start_h + crop_h, start_w:start_w + crop_w]

            # Resize back to original size
            resized_cropped_img = cv2.resize(cropped_img, (w, h), interpolation=cv2.INTER_LINEAR)

            # Append both original and modified image

            new_images.append(resized_cropped_img)  # Cropped and resized

        # Replace the original images with the new ones
        for resize_img in new_images:
            if not isinstance(resize_img, np.ndarray):
                resize_img = np.array(resize_img)
            results['img'].append(resize_img)

        return results

@TRANSFORMS.register_module()
class DTPVPack3DDetInputs(Pack3DDetInputs):
    INPUTS_KEYS = ['points', 'img', 'point_l', 'point_h']
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels', 'depths', 'centers_2d'
    ]
    INSTANCEDATA_2D_KEYS = [
        'gt_bboxes',
        'gt_bboxes_labels',
    ]
    SEG_KEYS = [
        'gt_seg_map', 'pts_instance_mask_l', 'pts_semantic_mask_l',
        'gt_semantic_seg', 'pts_instance_mask_h', 'pts_semantic_mask_h',
        'pts_instance_mask', 'pts_semantic_mask'
    ]

    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
        # Format 3D data
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor

        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = np.stack(results['img'], axis=0)
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
                else:
                    imgs = to_tensor(
                        np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                results['img'] = imgs
            else:
                img = results['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                # To improve the computational speed by by 3-5 times, apply:
                # `torch.permute()` rather than `np.transpose()`.
                # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
                # for more details
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(
                        np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['img'] = img

        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_bboxes_labels', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers_2d', 'depths', 'gt_labels_3d',
                'pts_instance_mask_h', 'pts_semantic_mask_h', 'pts_instance_mask_l', 'pts_semantic_mask_l'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])
        if 'gt_bboxes_3d' in results:
            if not isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = to_tensor(results['gt_bboxes_3d'])

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = to_tensor(
                results['gt_semantic_seg'][None])
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][None, ...]

        data_sample = Det3DDataSample()
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()
        gt_pts_seg = PointData()

        data_metas = {}
        for key in self.meta_keys:
            if key in results:
                data_metas[key] = results[key]
            elif 'images' in results:
                if len(results['images'].keys()) == 1:
                    cam_type = list(results['images'].keys())[0]
                    # single-view image
                    if key in results['images'][cam_type]:
                        data_metas[key] = results['images'][cam_type][key]
                else:
                    # multi-view image
                    img_metas = []
                    cam_types = list(results['images'].keys())
                    for cam_type in cam_types:
                        if key in results['images'][cam_type]:
                            img_metas.append(results['images'][cam_type][key])
                    if len(img_metas) > 0:
                        data_metas[key] = img_metas
            elif 'lidar_points' in results:
                if key in results['lidar_points']:
                    data_metas[key] = results['lidar_points'][key]
        data_sample.set_metainfo(data_metas)

        inputs = {}
        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key == 'gt_bboxes_labels':
                        gt_instances['labels'] = results[key]
                    else:
                        gt_instances[self._remove_prefix(key)] = results[key]
                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                else:
                    raise NotImplementedError(f'Please modified '
                                              f'`Pack3DDetInputs` '
                                              f'to put {key} to '
                                              f'corresponding field')



        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg
        if 'eval_ann_info' in results:
            data_sample.eval_ann_info = results['eval_ann_info']
        else:
            data_sample.eval_ann_info = None

        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs

        return packed_results

