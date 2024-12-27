import json
import os
import pickle
from pathlib import Path
data_path = "./data/nuscenes"
DATA = Path(__file__).parent.parent / data_path

is_train = False
dataset_type = "train" if is_train else "val"

def simplify_pkl(pkl_path: Path, new_pkl_path: Path, all_front_camera_names: set):
    new_info = []
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        infos = data["infos"]
        for info in infos:
            image_name = info['cams']['CAM_FRONT']['data_path'].split('/')[-1]
            if image_name in all_front_camera_names:
                new_info.append(info)
        data.update({"infos": new_info})

    with open(new_pkl_path, "wb") as f:
        pickle.dump(data, f)

def show_pkl(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        infos = data["infos"]
        print(len(infos))
        info = infos[0]
        print(info.keys())
        cam_front_path = info['cams']['CAM_FRONT']['data_path']
        print(cam_front_path)

def get_all_front_camera_names(images_path:Path) -> set:
    all_front_camera_names = set()
    _p = images_path.glob("*.jpg")
    for img in _p:
        print(img.name)
        all_front_camera_names.add(img.name)
    return all_front_camera_names


def reduce_pkl(pkl_path: Path, new_pkl_path: Path, max_size=100) -> Path:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        infos = data["data_list"]
        data.update({"data_list": infos[:max_size]})
    with open(new_pkl_path, "wb") as f:
        pickle.dump(data, f)

# 'sample_idx', 'token', 'timestamp', 'ego2global', 'images', 'lidar_points', 'instances', 'pts_semantic_mask_path', 'cam_instances'
# lidar_path', 'token', 'sweeps', 'cams', 'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp', 'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts
if __name__ == "__main__":
    pkl_path = DATA / f"nuscenes_infos_{dataset_type}_all.pkl"
    new_pkl_path = DATA / f"nuscenes_infos_{dataset_type}.pkl"
    # simplify_pkl(pkl_path, mini_pkl_path)
    # show_pkl(pkl_path)
    # images_path = DATA / "samples" / "CAM_FRONT"
    # all_images = get_all_front_camera_names(images_path=images_path)
    # simplify_pkl(pkl_path=pkl_path, new_pkl_path=new_pkl_path, all_front_camera_names=all_images)
    reduce_pkl(pkl_path, new_pkl_path, max_size=50)

