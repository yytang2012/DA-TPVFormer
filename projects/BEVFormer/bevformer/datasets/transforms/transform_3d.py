import numpy as np
from numpy import random
import mmcv
from mmdet3d.registry import TRANSFORMS
from mmcv.transforms.base import BaseTransform


@TRANSFORMS.register_module()
class PadMultiViewImage(BaseTransform):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.

    Required Keys:
        - img (List[np.ndarray])

    Added Keys:
        - img (List[np.ndarray]) (updated)
        - img_shape (List[tuple])
        - pad_shape (List[tuple])
        - pad_fixed_size (tuple)
        - pad_size_divisor (int)

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def transform(self, results: dict) -> dict:
        """Call function to pad images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]

        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@TRANSFORMS.register_module()
class NormalizeMultiviewImage(BaseTransform):
    """Normalize the image.

    Required Keys:
        - img (List[np.ndarray])

    Added Keys:
        - img (List[np.ndarray]) (normalized)
        - img_norm_cfg (dict)

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        super().__init__()
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def transform(self, results: dict) -> dict:
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
                          for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage(BaseTransform):
    """Apply photometric distortion to image sequentially.

    Required Keys:
        - img (List[np.ndarray])

    Added Keys:
        - img (List[np.ndarray]) (distorted)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        super().__init__()
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def transform(self, results: dict) -> dict:
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,' \
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                       self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                              self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@TRANSFORMS.register_module()
class CustomCollect3D(BaseTransform):
    """Pack the inputs data for training/testing.

    Required Keys:
        Based on keys parameter

    Added Keys:
        - data (dict)
        - data_samples (dict)

    Args:
        keys (list[str]): Keys of results to be packed.
        meta_keys (list[str]): Meta keys to be packed into data_samples.
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'lidar2cam', 'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx',
                            'next_idx', 'pcd_scale_factor', 'pcd_rotation',
                            'pts_filename', 'transformation_3d_flow', 'scene_token',
                            'can_bus')):
        super().__init__()
        self.keys = keys
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains
                - 'inputs' (dict): The forward data of models
                - 'data_samples' (obj:`Det3DDataSample`): The annotation info of the sample
        """
        # Pack input data
        packed_results = dict()
        if 'img' in results:
            packed_results['inputs'] = dict(imgs=results['img'])
        else:
            packed_results['inputs'] = dict()

        # Pack data_samples
        data_samples = dict()
        for key in self.meta_keys:
            if key in results:
                data_samples[key] = results[key]

        for key in self.keys:
            if key in results:
                data_samples[key] = results[key]

        packed_results['data_samples'] = data_samples

        return packed_results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, '
        repr_str += f'meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class RandomScaleImageMultiViewImage(BaseTransform):
    """Random scale the image.

    Required Keys:
        - img (List[np.ndarray])
        - lidar2img (List[np.ndarray])

    Added Keys:
        - img (List[np.ndarray]) (scaled)
        - lidar2img (List[np.ndarray]) (updated)
        - img_shape (List[tuple])
        - ori_shape (List[tuple])

    Args:
        scales (list[float]): List of scales.
    """

    def __init__(self, scales=[]):
        super().__init__()
        self.scales = scales
        assert len(self.scales) == 1

    def transform(self, results: dict) -> dict:
        """Apply the transform.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]

        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale

        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]),
                                        return_scale=False) for idx, img in enumerate(results['img'])]

        # TODO: fix the following
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(scales={self.scales})'
        return repr_str