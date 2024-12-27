# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, List, Optional, Union
import copy
import random
import numpy as np
import torch
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion


@DATASETS.register_module()
class NuScenesTempralDataset(NuScenesDataset):
    """NuScenes Dataset with temporal support.

    This dataset adds temporal support and maintains compatibility with
    camera intrinsics and extrinsics processing.
    """

    def __init__(self,
                 queue_length: int = 4,
                 bev_size: tuple = (200, 200),
                 overlap_test: bool = False,
                 data_root: Optional[str] = None,
                 use_can_bus=False,
                 **kwargs) -> None:
        super().__init__(data_root, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.use_can_bus = use_can_bus
        # 初始化CAN bus
        if self.use_can_bus is True:
            self.nusc_can_bus = NuScenesCanBus(dataroot=data_root)
        else:
            self.nusc_can_bus = None

    # TODO: to fix the bug
    def get_data_info(self, index: int) -> Union[Dict, None]:
        """Get data info according to index."""
        info = super().get_data_info(index)
        if info is None:
            return None

        # 只需保留必要的信息组织，不需要重复计算
        input_dict = dict(
            sample_idx=info['token'],
            timestamp=info['timestamp'],
            scene_token=info['scene_token'],
            prev_idx=info.get('prev', None),
            next_idx=info.get('next', None),
            frame_idx=info.get('frame_idx', 0),
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation']
        )

        # CAN bus信息直接使用
        if self.use_can_bus is True:
            input_dict['can_bus'] = info['can_bus']  # 直接使用已有的can_bus数据

        # 相机信息直接使用已经计算好的转换
        if self.modality['use_camera']:
            input_dict.update(
                dict(
                    img_path=[cam_info['img_path'] for cam_type, cam_info in info['images'].items()],
                    lidar2img=[cam_info['lidar2img'] for cam_type, cam_info in info['images'].items()],
                    cam_intrinsic=[cam_info['cam2img'] for cam_type, cam_info in info['images'].items()],
                    lidar2cam=[cam_info['lidar2cam'] for cam_type, cam_info in info['images'].items()]
                ))

        # Process ego pose and can_bus only if not already processed
        if 'can_bus' in input_dict and not isinstance(input_dict['can_bus'], np.ndarray):
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

        return input_dict

    def prepare_data(self, index: int) -> Union[Dict, None]:
        """Prepare data for training or testing."""
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        example = self.pipeline(input_dict)
        if self.test_mode:
            return example

        # Filter empty ground truth
        if (not self.test_mode) and (example is None or
                                     len(example['data_samples'].gt_instances_3d) == 0):
            return None

        return example

    def prepare_temporal_data(self, index: int) -> Union[Dict, None]:
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

    def union2one(self, queue: List[Dict]) -> Dict:
        """Unite temporal frames into one single data dict."""
        result_dict = queue[-1]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None

        # Collect image tensors
        imgs_list = [each['inputs'] for each in queue]
        result_dict['inputs'] = torch.stack(imgs_list)

        # Process each frame's meta information
        for i, each in enumerate(queue):
            metas_map[i] = each['data_samples'].metainfo

            # Handle scene transitions and update temporal information
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        result_dict['data_samples'].metainfo.update(metas_map)
        return result_dict

    def __getitem__(self, idx: int) -> Dict:
        """Get item from dataset."""
        if self.test_mode:
            return self.prepare_data(idx)

        while True:
            data = self.prepare_temporal_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
