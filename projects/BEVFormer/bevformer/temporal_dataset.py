# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
from typing import Dict, List, Optional, Union

import numpy as np
from mmengine import print_log
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class NuScenesTemporalDataset(NuScenesDataset):
    """NuScenes Dataset with temporal support.

    This dataset adds temporal support and maintains compatibility with
    camera intrinsics and extrinsics processing.

    Args:
        queue_length (int): Length of frame sequence. Defaults to 4.
        overlap_test (bool): Whether to use overlap in testing. Defaults to False.
        data_root (str, optional): Data root path. Defaults to None.
        use_can_bus (bool): Whether to use CAN bus data. Defaults to False.
        **kwargs: Other arguments passed to parent class.
    """

    def __init__(self,
                 queue_length: int = 4,
                 overlap_test: bool = False,
                 data_root: Optional[str] = None,
                 use_can_bus: bool = False,
                 **kwargs) -> None:
        super().__init__(data_root, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.use_can_bus = use_can_bus
        self.nusc_can_bus = NuScenesCanBus(dataroot=data_root) if use_can_bus else None

    def _process_can_bus(self, input_dict: Dict) -> np.ndarray:
        """Process CAN bus data.

        Args:
            input_dict (Dict): Input dict containing CAN bus data.

        Returns:
            np.ndarray: Processed CAN bus data array.
        """
        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = np.zeros(18) if input_dict['can_bus'] is None else input_dict['can_bus']

        # Fill translation and rotation
        can_bus[:3] = translation
        can_bus[3:7] = rotation

        # Calculate patch angle
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360

        # Fill angles
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return can_bus

    def prepare_data(self, index: int) -> Union[Dict, None]:
        """Prepare data for training or testing.

        Args:
            index (int): Data index.

        Returns:
            Union[Dict, None]: Prepared data dict or None if invalid.
        """
        # Get original data info
        ori_input_dict = self.get_data_info(index)
        input_dict = copy.deepcopy(ori_input_dict)

        # Add basic info
        input_dict.update({
            'box_type_3d': self.box_type_3d,
            'box_mode_3d': self.box_mode_3d,
            'prev_idx': ori_input_dict.get('prev', None),
            'next_idx': ori_input_dict.get('next', None),
            'frame_idx': ori_input_dict.get('frame_idx', 0)
        })

        # Process CAN bus data if needed
        if self.use_can_bus:
            input_dict['can_bus'] = self._process_can_bus(ori_input_dict)

        # Add camera data if using camera modality
        if self.modality['use_camera']:
            cam_info_dict = {
                'img_path': [info['img_path'] for _, info in ori_input_dict['images'].items()],
                'lidar2img': [info['lidar2img'] for _, info in ori_input_dict['images'].items()],
                'cam_intrinsic': [info['cam2img'] for _, info in ori_input_dict['images'].items()],
                'lidar2cam': [info['lidar2cam'] for _, info in ori_input_dict['images'].items()]
            }
            input_dict.update(cam_info_dict)

        # Filter empty ground truth in training
        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                return None

        # Process through pipeline
        example = self.pipeline(input_dict)

        # Post-process filtering
        if not self.test_mode and self.filter_empty_gt:
            if example is None or len(example['data_samples'].gt_instances_3d.labels_3d) == 0:
                return None

        # Show instance variation if needed
        if self.show_ins_var and 'ann_info' in ori_input_dict:
            self._show_ins_var(
                ori_input_dict['ann_info']['gt_labels_3d'],
                example['data_samples'].gt_instances_3d.labels_3d)

        return example

    def prepare_temporal_data(self, index: int) -> Optional[List[dict]]:
        """Prepare temporal sequence data.

        Args:
            index (int): Current frame index.

        Returns:
            Optional[List[dict]]: List of prepared temporal data or None if invalid.
        """
        queue = []
        index_list = list(range(index - self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)

        for i in index_list:
            i = max(0, i)
            example = self.prepare_data(i)
            if example is None:
                return None
            queue.append(example)

        return self.union2one(queue)

    def union2one(self, queue: List[Dict]) -> List[dict]:
        """Unite temporal frames into one data dict.

        Args:
            queue (List[Dict]): List of temporal frame data.

        Returns:
            List[dict]: United temporal data.
        """
        prev_scene_token = None
        prev_pos = None
        prev_angle = None

        for i, each in enumerate(queue):
            _meta = each['data_samples'].metainfo

            if _meta['scene_token'] != prev_scene_token:
                # New scene starts
                _meta['prev_bev_exists'] = False
                prev_scene_token = _meta['scene_token']
                prev_pos = copy.deepcopy(_meta['can_bus'][:3])
                prev_angle = copy.deepcopy(_meta['can_bus'][-1])
                _meta['can_bus'][:3] = 0
                _meta['can_bus'][-1] = 0
            else:
                # Continue in same scene
                _meta['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(_meta['can_bus'][:3])
                tmp_angle = copy.deepcopy(_meta['can_bus'][-1])
                _meta['can_bus'][:3] -= prev_pos
                _meta['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

            each['data_samples'].set_metainfo(_meta)

        return queue

    def __getitem__(self, idx: int) -> Union[dict, None, List[dict]]:
        """Get item from dataset.

        Args:
            idx (int): Index of data.

        Returns:
            Union[dict, None, List[dict]]: Data item.
        """
        if self.test_mode:
            return self.prepare_data(idx)

        while True:
            data = self.prepare_temporal_data(idx)
            if data is None:
                print_log(
                    f"Failed to load data at index {idx}. This may be due to empty "
                    f"ground truth or invalid data. Retrying with new random index.",
                    logger='current'
                )
                idx = self._rand_another()
                continue
            return data