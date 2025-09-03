import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

from Zagreus.envs.base.learnable_controller import ButterworthFilter


if __name__ == "__main__":
    device = "cpu"
    num_envs = 1
    dt = 0.01  # 采样间隔 10ms
    cutoff_hz = 2.0  # 截止频率 2Hz

    # 初始化滤波器
    bw_filter = ButterworthFilter(num_envs=num_envs, dt=dt, cutoff_hz=cutoff_hz, device=device)

    # 生成模拟输入信号：低频正弦 + 高频噪声
    t = torch.arange(0, 5, dt)  # 5秒
    low_freq_signal = torch.sin(2 * math.pi * 1.0 * t)  # 1Hz 正弦
    high_freq_noise = 0.5 * torch.randn_like(t)  # 高频噪声
    input_signal = (low_freq_signal + high_freq_noise).unsqueeze(1)  # shape: (N, 1)
    
    # 由于 filter 定义的是 (num_envs, 3)，我们可以扩展到3维
    input_signal_3d = input_signal.repeat(1, 3)  # shape: (N, 3)

    # 存储输出
    output_signal = []

    # 遍历每个时间步
    for x in input_signal_3d:
        y = bw_filter(x.unsqueeze(0))  # 添加 batch 维度
        output_signal.append(y.squeeze(0).detach())

    output_signal = torch.stack(output_signal)

    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(t, input_signal_3d[:, 0], label="Input Signal (noisy)")
    plt.plot(t, output_signal[:, 0], label="Filtered Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Butterworth Filter Input vs Output")
    plt.legend()
    plt.grid(True)
    plt.show()