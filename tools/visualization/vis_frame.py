import argparse
import os
import shutil
import torch
import numpy as np
from mmengine import Config

from mmdet3d.registry import DATASETS, MODELS
from mmengine.registry import init_default_scope
from mayavi import mlab
from collections import OrderedDict
from mmengine.dataset import default_collate
from mmdet3d.structures import Det3DDataSample
from torch import Tensor

def revise_ckpt(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if tmp_k.startswith('module.'):
        return OrderedDict({k[7:]: v for k, v in state_dict.items()})
    return state_dict


def get_grid_coords(dims, resolution):
    g_xx = np.arange(0, dims[0])
    g_yy = np.arange(0, dims[1])
    g_zz = np.arange(0, dims[2])
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])
    return (coords_grid * resolution) + resolution / 2


def draw(voxels, pred_pts, vox_origin, voxel_size, grid, pt_label, save_dir, cam_positions=None, focal_positions=None, timestamp=None, mode=0):
    w, h, z = voxels.shape
    grid = grid.astype(np.int32)
    grid_coords = get_grid_coords([w, h, z], voxel_size) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    if mode == 0:
        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    elif mode == 1:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        pred_pts = pred_pts[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, pred_pts.reshape(-1)]).T
    elif mode == 2:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        gt_label = pt_label[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, gt_label.reshape(-1)]).T
    else:
        raise NotImplementedError

    grid_coords[grid_coords[:, 3] == 17, 3] = 20
    fov_voxels = grid_coords[(grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 20)]

    figure = mlab.figure(size=(1920, 1080), bgcolor=(1, 1, 1))
    voxel_size = sum(voxel_size) / 3
    plt_plot = mlab.points3d(
        fov_voxels[:, 1], fov_voxels[:, 0], fov_voxels[:, 2],
        fov_voxels[:, 3], colormap="viridis", scale_factor=0.95 * voxel_size,
        mode="cube", opacity=1.0, vmin=1, vmax=19
    )

    colors = np.array([
        [255, 120,  50, 255], [255, 192, 203, 255], [255, 255,   0, 255], [0, 150, 245, 255],
        [0, 255, 255, 255], [255, 127,   0, 255], [255,   0,   0, 255], [255, 240, 150, 255],
        [135,  60,   0, 255], [160,  32, 240, 255], [255,   0, 255, 255], [139, 137, 137, 255],
        [75,   0,  75, 255], [150, 240,  80, 255], [230, 230, 250, 255], [0, 175,   0, 255],
        [0, 255, 127, 255], [255,  99,  71, 255], [0, 191, 255, 255]
    ]).astype(np.uint8)

    plt_plot.glyph.scale_mode = "scale_by_vector"
    plt_plot.module_manager.scalar_lut_manager.lut.table = colors

    scene = figure.scene
    scene.camera.position = [0.751, -35.08, 16.71]
    scene.camera.focal_point = [0.751, -34.21, 16.21]
    scene.camera.view_angle = 40.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [0.01, 300.]
    scene.camera.compute_view_plane_normal()
    scene.render()
    mlab.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--ckpt-path', required=True)
    parser.add_argument('--save-path', type=str, default='out/vis')
    parser.add_argument('--frame-idx', type=int, nargs='+', default=[0])
    parser.add_argument('--mode', type=int, default=0, help='0: voxel pred, 1: point pred, 2: gt')
    args = parser.parse_args()

    cfg = Config.fromfile(args.py_config)
    device = torch.device('cuda:0')

    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    # 构建模型
    model = MODELS.build(cfg.model).to(device)

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(revise_ckpt(checkpoint))
    model.eval()

    # 构建验证集（使用 val_dataloader.dataset）
    dataset_cfg = cfg.val_dataloader['dataset']
    dataset = DATASETS.build(dataset_cfg)

    # 体素设置
    voxel_origin = cfg.point_cloud_range[:3]
    voxel_max = cfg.point_cloud_range[3:]
    grid_size = cfg.get('grid_shape') or cfg.get('grid_size')
    resolution = [(e - s) / l for e, s, l in zip(voxel_max, voxel_origin, grid_size)]

    for index in args.frame_idx:
        print(f'Visualizing frame {index}')
        sample = dataset[index]
        if 'data_samples' in sample:
            data_sample = sample['data_samples']
            if isinstance(data_sample, Det3DDataSample):
                sample['data_samples'] = [data_sample]
        if 'inputs' in sample:
            inputs = sample['inputs']
            if 'img' in inputs:
                imgs = inputs['img']
                if isinstance(imgs, Tensor):
                    sample['inputs']['img'] = [imgs]
            if 'points' in inputs:
                points = inputs['points']
                if isinstance(points, Tensor):
                    sample['inputs']['points'] = [points]

        data_sample = sample['data_samples']
        inputs = sample['inputs']
        # 获取图像和点云
        imgs = sample['inputs']['img']  # Tensor, shape: (N_cam, 3, H, W)
        points = sample['inputs']['points']  # Tensor, shape: (N_pts, 3)

        # 获取数据样本元信息
        img_metas = data_sample[0].metainfo  # dict，包含相机姿态、timestamp 等
        timestamp = img_metas.get('timestamp', index)  # 有些配置没有 timestamp，就用 index 代替


        # 获取标签
        pt_label = data_sample[0].gt_pts_seg.pts_semantic_mask  # shape: (N_pts,)
        # voxel label、grid 可以从数据预处理管道中间结果保存或手动计算（见下）

        # ⚠️ MMDetection3D 默认不会直接返回 voxel grid，需要你在 dataset 或 pipeline 中手动加

        # 转移到 device
        # imgs = imgs.float().unsqueeze(0).to(device)  # [1, N_cam, 3, H, W]
        # points = points.unsqueeze(0).to(device)  # [1, N_pts, 3]
        # sample['inputs']['imgs'] = imgs

        # 推理
        with torch.no_grad():
            # outputs_vox, outputs_pts = model.test_step(sample)
            outputs_pts = model.test_step(sample            # predict_vox = torch.argmax(outputs_vox, dim=1).squeeze(0).cpu().numpy()
            predict_pts = torch.argmax(outputs_pts[0].pts_seg_logits, dim=1).squeeze().cpu().numpy()

        # 保存图像等
        frame_dir = os.path.join(args.save_path, str(index))
        os.makedirs(frame_dir, exist_ok=True)

        # 可选：如果你需要 filelist，但现在没直接返回，你得修改 Dataset 的 get_data_info() 加载 filelist 信息

        # 调用绘图函数
        draw(
            predict_vox,
            predict_pts,
            voxel_origin,
            resolution,
            points.squeeze(0).cpu().numpy(),  # grid 用点云代替（如果你没有单独 grid）
            pt_label.squeeze(-1).cpu().numpy(),
            frame_dir,
            img_metas.get('cam_positions', None),
            img_metas.get('focal_positions', None),
            timestamp,
            mode=args.mode
        )

