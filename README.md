<div align="center">   
  
# Distance-Aware Tri-Perspective View for Efficient 3D Perception in Autonomous Driving
</div>


### [Paper](https://arxiv.org/pdf/2302.07817) | [Project Page](https://wzzheng.net/TPVFormer/) | [Leaderboard](https://www.nuscenes.org/lidar-segmentation?externalData=all&mapData=all&modalities=Camera)



> Yutao Tang*, Jigang Zhao\* , Zhengrui Qin, Lingying Zhao, Guangxi Chen$\ddagger$,Jie Ren$\ddagger$

\* Equal contribution  $\ddagger$ Corresponding author




# Abstract
Three-dimensional environmental perception remains a critical bottleneck in autonomous driving, where existing vision-based dense representations face an intractable trade-off between spatial resolution and computational complexity. Current methods, including Bird's Eye View (BEV) and Tri-Perspective View (TPV), apply uniform perception precision across all spatial regions, disregarding the fundamental safety principle that near-field objects demand high-precision detection for collision avoidance while distant objects permit lower initial accuracy. This uniform treatment squanders computational resources and constrains real-time deployment. We introduce Distance-Aware Tri-Perspective View (DA-TPV), a novel framework that allocates computational resources proportional to operational risk. DA-TPV employs a hierarchical dual-plane architecture for each viewing direction: low-resolution planes capture global scene context while high-resolution planes deliver fine-grained perception within safety-critical reaction zones. Through distance-adaptive feature fusion, our method dynamically concentrates processing power where it most directly impacts vehicle safety. Extensive experiments on nuScenes demonstrate that DA-TPV matches or exceeds single high-resolution TPV performance while reducing memory consumption by 26.3\% and achieving real-time inference. This work establishes distance-aware perception as a practical paradigm for deploying sophisticated three-dimensional understanding within automotive computational constraints. 


# Methods
![method](figs/arch.png "model arch")


<!-- # Getting Started
- [Installation](docs/install.md) 
- [Prepare Dataset](docs/prepare_dataset.md)
- [Run and Eval](docs/getting_started.md) -->

# Model Zoo
## Nuscenes datasets
| Method | mIoU  | Download |
| :-- | :--: |  :-- |
| tpv (200×200×16) | — | [model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_fp16_epoch_24.pth) / [log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_fp16_epoch_24.log) |
| tpv (100×100×8) | — | [model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth) / [log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.log) |
| datpv (100×100×8) | — | [model](https://pan.baidu.com/s/1J-dD0btopLW4oY1p-Gn-RA 提取码: cbqd) / [log](https://pan.baidu.com/s/191TdQAtjcBZep6zrm16XZQ 提取码: wdeu) |



| R50 · BEVFormerV2-t2 · 48ep | — | [config](projects/configs/bevformerv2/bevformerv2-r50-t2-48ep.py) | [model/log](https://drive.google.com/drive/folders/1bSyuFWxfJSIidGV7bC8jx2NR7idRN9-s?usp=sharing) |
| R50 · BEVFormerV2-t8 · 24ep | — | [config](projects/configs/bevformerv2/bevformerv2-r50-t8-24ep.py) | [model/log](https://drive.google.com/drive/folders/1Ml_usx5BNx43CFH1Di2OTazuzSyAlBto?usp=sharing) |





# Acknowledgement

Many thanks to these excellent open source projects:
- [tpvformer](https://github.com/wzzheng/TPVFormer) 
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
