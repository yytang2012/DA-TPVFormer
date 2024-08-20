import torch
import pprint


def analyze_pretrained_weights(checkpoint_path, prefix=''):
    # 加载预训练权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 创建一个字典来存储每一层的信息
    layer_info = {}

    for key, value in state_dict.items():
        if key.startswith(prefix):
            key = key[len(prefix):]  # 移除前缀

        # 分割键名以获取层级结构
        parts = key.split('.')
        current_dict = layer_info
        for part in parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]

        # 存储张量的形状和类型
        current_dict[parts[-1]] = {
            'shape': list(value.shape),
            'dtype': str(value.dtype)
        }

    # 打印层级结构
    print("预训练权重的结构:")
    pprint.pprint(layer_info, width=120, compact=True)

    # 打印一些统计信息
    total_params = sum(param.numel() for param in state_dict.values())
    print(f"\n总参数数量: {total_params:,}")
    print(f"层数: {len(state_dict)}")


# 使用脚本
checkpoint_path = '../checkpoints/tpvformer_pretrained_fcos3d_r101_dcn.pth'
prefix = ''
analyze_pretrained_weights(checkpoint_path, prefix)