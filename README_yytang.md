sudo apt install nvidia-cuda-toolkit

conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia

pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4' --no-cache-dir
mim install 'mmdet>=3.0.0' --no-cache-dir
pip install --no-cache-dir -e .



export RANK=0
python tools/test.py ./projects/TPVFormer/configs/tpvformer_8xb1-2x_nus-seg.py ./checkpoints/tpvformer_pretrained_fcos3d_r101_dcn.pth --launcher pytorch

python tools/train.py ./projects/TPVFormer/configs/tpvformer_8xb1-2x_nus-seg-small-mini-test.py  --launcher pytorch# 准备数据集
```shell
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

# Train TPVFormer on GTX 4090 GPU
```shell
bash tools/dist_train.sh projects/TPVFormer/configs/tpvformer_8xb1-2x_nus-seg-small.py 1
```

