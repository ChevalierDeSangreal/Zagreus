import os
import torch
from Zagreus.config import ROOT_DIR

# ------------------- 配置 -------------------
param_file = "worldVer2.pth"  # 你保存的模型文件
param_path = os.path.join(ROOT_DIR, "param", param_file)

# ------------------- 加载模型参数 -------------------
checkpoint = torch.load(param_path, map_location="cpu")  # 使用CPU加载即可

# ------------------- 打印内容 -------------------
print("Checkpoint keys:", checkpoint.keys())  # 包含 'epoch', 'model_state_dict', 'optimizer_state_dict', 'loss'

print("\nEpoch:", checkpoint.get('epoch', 'N/A'))
print("Saved loss:", checkpoint.get('loss', 'N/A'))

print("\nModel state dict:")
for name, tensor in checkpoint['model_state_dict'].items():
    print(f"\n{name}:")
    print(tensor)  # 打印张量的具体值


print("\nOptimizer state dict keys:")
for k in checkpoint['optimizer_state_dict'].keys():
    print(k)
