import isaacgym  # noqa
from isaacgym import gymutil
from isaacgym.torch_utils import *
import torch
import math

class ButterworthFilter:
    def __init__(self, num_envs, dt, cutoff_hz, device="cpu", dtype=torch.float32):
        """
        Second-order Butterworth low-pass filter.
        Args:
            num_envs (int): Number of parallel environments.
            dt (float): Sampling period.
            cutoff_hz (float): Cutoff frequency (Hz).
            device (str): Torch device.
            dtype (torch.dtype): Torch data type.
        """
        self.num_envs = num_envs
        self.dt = dt
        self.cutoff_hz = cutoff_hz
        self.device = device
        self.dtype = dtype

        # Compute filter coefficients
        self._compute_coeffs()

        # Initialize previous input and output states
        self._x1 = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self._x2 = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self._y1 = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self._y2 = torch.zeros(num_envs, 3, device=device, dtype=dtype)

    def _compute_coeffs(self):
        omega_c = 2 * math.pi * self.cutoff_hz
        tan_wc = math.tan(omega_c * self.dt / 2)
        sqrt2 = math.sqrt(2)

        a0 = 1 + sqrt2 * tan_wc + tan_wc * tan_wc
        self.b0 = (tan_wc * tan_wc) / a0
        self.b1 = 2 * self.b0
        self.b2 = self.b0
        self.a1 = 2 * (tan_wc * tan_wc - 1) / a0
        self.a2 = (1 - sqrt2 * tan_wc + tan_wc * tan_wc) / a0

    def reset(self, env_ids=None):
        """
        Reset the filter states.
        Args:
            env_ids (list or torch.Tensor or None): Indices of environments to reset. If None, reset all.
        """
        if env_ids is None:
            self._x1.zero_()
            self._x2.zero_()
            self._y1.zero_()
            self._y2.zero_()
        else:
            self._x1[env_ids].zero_()
            self._x2[env_ids].zero_()
            self._y1[env_ids].zero_()
            self._y2[env_ids].zero_()

    def __call__(self, x):
        """
        Apply the second-order Butterworth low-pass filter to the input.
        Args:
            x (torch.Tensor): Input tensor of shape (num_envs, 3)
        Returns:
            y (torch.Tensor): Filtered output tensor of shape (num_envs, 3)
        """
        y = (
            self.b0 * x +
            self.b1 * self._x1 +
            self.b2 * self._x2 -
            self.a1 * self._y1 -
            self.a2 * self._y2
        )
        # Update history
        self._x2 = self._x1.clone()
        self._x1 = x.clone()
        self._y2 = self._y1.clone()
        self._y1 = y.clone()
        return y

rates = torch.load("/home/core/wangzimo/Zagreus/Zagreus/data/all_rates.pt")   # 加载
filter = ButterworthFilter(num_envs=8, dt=0.001, cutoff_hz=40, device="cuda:0")

for idx, rate in enumerate(rates):
    # print("Input rate:", rate)
    print("Max of input rate:", rate.max())
    filtered_rate = filter(rate)
    # print("Filtered rate:", filtered_rate)
    if idx > 15:
        break
    if torch.isnan(filtered_rate).any():
        print("NaN detected in filtered rate")
        break
