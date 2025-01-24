# Data Preparation for BEVFormer

This guide explains how to prepare the NuScenes dataset for BEVFormer training and evaluation.

## Download NuScenes Dataset

First, download the NuScenes dataset from the [official website](https://www.nuscenes.org/download). You can choose either:
- Full dataset (v1.0)
- Mini dataset (v1.0-mini)
- CAN bus expansion pack (optional, for BEVFormer with vehicle motion)

## Dataset Structure

After downloading and extracting, symlink the dataset root to `$BEVFormer/data`. The folder structure should be:

```
BEVFormer
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test           # for full dataset only
|   |   ├── v1.0-trainval       # for full dataset only
|   |   ├── v1.0-mini          # for mini dataset only
|   |   ├── can_bus (optional)
```

## Data Preprocessing

### Full Dataset

For standard preparation without CAN bus information:
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

With CAN bus information:
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes_canbus --with-canbus
```

### Mini Dataset

For standard preparation without CAN bus information:
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini
```

With CAN bus information:
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes_canbus --version v1.0-mini --with-canbus
```

After running these commands, the folder structure will be:

```
BEVFormer
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test          # for full dataset only
|   |   ├── v1.0-trainval      # for full dataset only
|   |   ├── v1.0-mini         # for mini dataset only
|   |   ├── can_bus (optional)
│   │   ├── nuscenes_canbus_infos_train.pkl
│   │   ├── nuscenes_canbus_infos_val.pkl
│   │   ├── nuscenes_canbus_infos_test.pkl    # for full dataset only
```

## Data Format

The prepared data includes `.pkl` files that contain all necessary information for training and evaluation. Each file contains:

### For Training/Validation Sets

- Basic scene info (timestamp, ego pose, calibration)
- Image data for 6 cameras (CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT)
- 3D bounding box annotations
- CAN bus information (if enabled)
  - Vehicle pose
  - Vehicle motion (velocity, acceleration)
  - Other sensor readings

### For Test Set (Full Dataset Only)

- Basic scene info
- Image data for 6 cameras
- CAN bus information (if enabled)
- No annotations

## Dataset Statistics

### Full Dataset (v1.0)
- Train set: 28,130 samples
- Validation set: 6,019 samples
- Test set: 6,008 samples
- Total scenes: 1000

### Mini Dataset (v1.0-mini)
- Train set: 3,976 samples
- Validation set: 252 samples
- Total scenes: 10

## Notes

1. BEVFormer uses `--extra-tag nuscenes_canbus` to generate dataset infos that include temporal information required for the temporal modules.

2. When using CAN bus data, make sure your configuration file properly handles the CAN bus information in the data pipeline.

3. If CAN bus data is not available even with `--with-canbus` flag, the script will automatically fall back to using zeros for motion information and print a warning.

4. For temporal feature learning in BEVFormer, the data loading pipeline is configured to handle sequences of frames. The `queue_length` parameter in your config file determines how many frames are loaded in sequence.

## Troubleshooting

If you encounter issues:

1. Verify the dataset structure matches the expected format
2. If using CAN bus data, ensure the expansion pack is properly extracted
3. Check the paths in your configuration file match your actual data location
4. Ensure you have enough disk space:
   - Full dataset processing: approximately 200GB
   - Mini dataset processing: approximately 25GB