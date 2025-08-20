import os
import random
import numpy as np

import isaacgym  # noqa
from isaacgym import gymutil
from isaacgym.torch_utils import *

import time
import pytz
from datetime import datetime
import argparse
# 导入自定义模块
from Zagreus.envs.base.dynamics_iris_learnable import IrisDynamics
from Zagreus.dataset import TrajDataset
from Zagreus.config import ROOT_DIR

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
device = 'cuda:0'
dynamics = IrisDynamics(device=device)
opt = torch.optim.Adam(dynamics.parameters(), lr=1e-2)

state = torch.zeros(1, 12, device=device)
action = torch.rand(1, 4, device=device)
dt = 0.02

out, _ = dynamics(action, state, dt)
loss = out.sum()
loss.backward()
print(dynamics.tau_rotate.grad)  # 如果不是 None，说明梯度传回来了
print(dynamics.kinv_ang_vel_tau.grad)  # 如果不是 None，说明梯度传回来了
