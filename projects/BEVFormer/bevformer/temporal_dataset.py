# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, List, Optional, Union
import copy
import random
import numpy as np
import torch
from mmengine import print_log
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes, Det3DDataSample
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion


@DATASETS.register_module()
class NuScenesTemporalDataset(NuScenesDataset):
    """NuScenes Dataset with temporal support.

    This dataset adds temporal support and maintains compatibility with
    camera intrinsics and extrinsics processing.
    """

    def __init__(self,
                 queue_length: int = 4,
                 # bev_size: tuple = (200, 200),
                 overlap_test: bool = False,
                 data_root: Optional[str] = None,
                 use_can_bus=False,
                 **kwargs) -> None:
        super().__init__(data_root, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        # self.bev_size = bev_size
        self.use_can_bus = use_can_bus
        # Initialize CAN bus
        if self.use_can_bus is True:
            self.nusc_can_bus = NuScenesCanBus(dataroot=data_root)
        else:
            self.nusc_can_bus = None

    def prepare_data(self, index: int) -> Union[Dict, None]:
        """Prepare data for training or testing."""

        ori_input_dict = self.get_data_info(index)
        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(ori_input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d

        input_dict.update({
            "prev_idx": ori_input_dict.get('prev', None),
            "next_idx": ori_input_dict.get('next', None),
            "frame_idx": ori_input_dict.get('frame_idx', 0), }
        )

        if self.use_can_bus is True:
            input_dict['can_bus'] = ori_input_dict['can_bus']

        if self.modality['use_camera']:
            input_dict.update(
                dict(
                    img_path=[cam_info['img_path'] for cam_type, cam_info in ori_input_dict['images'].items()],
                    lidar2img=[cam_info['lidar2img'] for cam_type, cam_info in ori_input_dict['images'].items()],
                    cam_intrinsic=[cam_info['cam2img'] for cam_type, cam_info in ori_input_dict['images'].items()],
                    lidar2cam=[cam_info['lidar2cam'] for cam_type, cam_info in ori_input_dict['images'].items()]
                ))

        # Process ego pose and can_bus only if not already processed
        if 'can_bus' in input_dict:
            rotation = Quaternion(input_dict['ego2global_rotation'])
            translation = input_dict['ego2global_translation']
            can_bus = np.zeros(18) if input_dict['can_bus'] is None else input_dict['can_bus']
            can_bus[:3] = translation
            can_bus[3:7] = rotation
            patch_angle = quaternion_yaw(rotation) / np.pi * 180
            if patch_angle < 0:
                patch_angle += 360
            can_bus[-2] = patch_angle / 180 * np.pi
            can_bus[-1] = patch_angle
            input_dict['can_bus'] = can_bus

        # pre-pipline return None to random another in `__getitem__`
        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                return None

        example = self.pipeline(input_dict)

        if not self.test_mode and self.filter_empty_gt:
            # after pipeline drop the example with empty annotations
            # return None to random another in `__getitem__`
            if example is None or len(
                    example['data_samples'].gt_instances_3d.labels_3d) == 0:
                return None

        if self.show_ins_var:
            if 'ann_info' in ori_input_dict:
                self._show_ins_var(
                    ori_input_dict['ann_info']['gt_labels_3d'],
                    example['data_samples'].gt_instances_3d.labels_3d)
            else:
                print_log(
                    "'ann_info' is not in the input dict. It's probably that "
                    'the data is not in training mode',
                    'current',
                    level=30)

        return example

    def prepare_temporal_data(self, index: int) -> List[dict]:
        """Prepare temporal data for training."""
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
        """Unite temporal frames into one single data dict."""
        prev_scene_token = None
        prev_pos = None
        prev_angle = None

        # Process each frame's meta information
        for i, each in enumerate(queue):
            _meta = each['data_samples'].metainfo

            # Handle scene transitions and update temporal information
            if _meta['scene_token'] != prev_scene_token:
                _meta['prev_bev_exists'] = False
                prev_scene_token = _meta['scene_token']
                prev_pos = copy.deepcopy(_meta['can_bus'][:3])
                prev_angle = copy.deepcopy(_meta['can_bus'][-1])
                _meta['can_bus'][:3] = 0
                _meta['can_bus'][-1] = 0
            else:
                _meta['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(_meta['can_bus'][:3])
                tmp_angle = copy.deepcopy(_meta['can_bus'][-1])
                _meta['can_bus'][:3] -= prev_pos
                _meta['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
            each['data_samples'].set_metainfo(_meta)
        return queue

    def __getitem__(self, idx: int) -> Optional[dict]:
        """Get item from dataset."""
        if self.test_mode:
            return self.prepare_data(idx)

        while True:
            data = self.prepare_temporal_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
