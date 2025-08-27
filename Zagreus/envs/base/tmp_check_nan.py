import torch

# 创建一个包含0和正数的tensor
x = torch.tensor([0.0, 0.25, 1.0], requires_grad=True)

# 正向传播
y = torch.sqrt(x)

# 输出正向结果
print("Forward sqrt(x):", y)

# 反向传播
try:
    y.sum().backward()
    print("Gradients:", x.grad)
except Exception as e:
    print("Backward error:", e)
