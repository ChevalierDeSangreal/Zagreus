import torch
import matplotlib.pyplot as plt

# 定义函数: f(x) = (x^n) * (1e- n)
def f(x, n):
    return (x**n) * (10**(-n))

# 设置范围
x_vals = torch.linspace(1.0, 5.0, 100)

results = []
grads = []

# 计算不同 n 下的前向和梯度
for n in [5, 10, 20, 50]:
    y_vals = []
    g_vals = []
    for x in x_vals:
        x = x.clone().requires_grad_(True)
        y = f(x, n)
        y.backward()
        y_vals.append(y.item())
        g_vals.append(x.grad.item())
    results.append((n, y_vals))
    grads.append((n, g_vals))

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 前向值
for n, y_vals in results:
    axes[0].plot(x_vals, y_vals, label=f"n={n}")
axes[0].set_title("Forward values: f(x) = (x^n)*1e-n")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].legend()
axes[0].grid(True)

# 梯度
for n, g_vals in grads:
    axes[1].plot(x_vals, g_vals, label=f"n={n}")
axes[1].set_title("Gradient wrt x")
axes[1].set_xlabel("x")
axes[1].set_ylabel("df/dx")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
