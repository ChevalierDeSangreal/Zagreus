# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from torch import Tensor
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_quaternion, axis_angle_to_matrix
import math

@torch.jit.script
def compute_vee_map(skew_matrix):
    # type: (Tensor) -> Tensor

    # return vee map of skew matrix
    vee_map = torch.stack(
        [-skew_matrix[:, 1, 2], skew_matrix[:, 0, 2], -skew_matrix[:, 0, 1]], dim=1)
    return vee_map

def rand_circle_point(batch_size, r, device):
    thetas = torch.rand(batch_size, device=device) * 2 * torch.tensor(np.pi, device=device)
    x = r * torch.cos(thetas)
    y = r * torch.sin(thetas)
    # print(torch.stack([x, y], dim=1), torch.initial_seed())
    return torch.stack([x, y], dim=1)

def set_circle_point(batch_size, r, device, thetas):
    """
    Generate a position on the circle
    r is the radius of circle
    thetas is the direction
    thetas should be [0, 2pi]
    """
    # thetas = torch.rand(batch_size, device=device) * 2 * torch.tensor(np.pi, device=device)
    thetas = thetas * torch.ones(batch_size, device=device) /180 * torch.tensor(np.pi, device=device)
    x = r * torch.cos(thetas)
    y = r * torch.sin(thetas)
    # print(torch.stack([x, y], dim=1), torch.initial_seed())
    return torch.stack([x, y], dim=1)

# if __name__ == "__main__":
#     print(rand_circle_point(2, 5, 'cuda:0'))

def euler2qua(euler):
	rotation_matrices = euler_angles_to_matrix(euler, "ZYX")
	qua = matrix_to_quaternion(rotation_matrices)[:, [3, 2, 1, 0]]
	return qua


def ned_to_enu_xyz(xyz_ned: torch.Tensor) -> torch.Tensor:
    """
    NED坐标系的xyz转换到ENU坐标系
    输入和输出shape均为 (batch_size, 3)
    公式：
        x_enu = y_ned
        y_enu = x_ned
        z_enu = -z_ned
    """
    x_ned = xyz_ned[:, 0]
    y_ned = xyz_ned[:, 1]
    z_ned = xyz_ned[:, 2]
    x_enu = y_ned
    y_enu = x_ned
    z_enu = -z_ned
    return torch.stack([x_enu, y_enu, z_enu], dim=1)


def enu_to_ned_xyz(xyz_enu: torch.Tensor) -> torch.Tensor:
    """
    ENU坐标系的xyz转换到NED坐标系
    输入和输出shape均为 (batch_size, 3)
    公式：
        x_ned = y_enu
        y_ned = x_enu
        z_ned = -z_enu
    """
    x_enu = xyz_enu[:, 0]
    y_enu = xyz_enu[:, 1]
    z_enu = xyz_enu[:, 2]
    x_ned = y_enu
    y_ned = x_enu
    z_ned = -z_enu
    return torch.stack([x_ned, y_ned, z_ned], dim=1)


def ned_to_enu_euler(euler_ned: torch.Tensor) -> torch.Tensor:
    """
    NED欧拉角（roll, pitch, yaw）转换到ENU定义
    输入和输出shape均为 (batch_size, 3)
    输入和输出单位均为弧度
    变换关系：
        yaw_enu = pi/2 - yaw_ned
        pitch_enu = -pitch_ned
        roll_enu = roll_ned
    """
    roll_ned = euler_ned[:, 0]
    pitch_ned = euler_ned[:, 1]
    yaw_ned = euler_ned[:, 2]

    yaw_enu = (math.pi / 2) - yaw_ned
    # 对yaw取模到[-pi, pi]区间，保持连续性（可选）
    yaw_enu = (yaw_enu + math.pi) % (2 * math.pi) - math.pi

    pitch_enu = -pitch_ned
    roll_enu = roll_ned

    return torch.stack([roll_enu, pitch_enu, yaw_enu], dim=1)


def enu_to_ned_euler(euler_enu: torch.Tensor) -> torch.Tensor:
    """
    ENU欧拉角（roll, pitch, yaw）转换到NED定义
    输入和输出shape均为 (batch_size, 3)
    输入和输出单位均为弧度
    变换关系：
        yaw_ned = pi/2 - yaw_enu
        pitch_ned = -pitch_enu
        roll_ned = roll_enu
    """
    roll_enu = euler_enu[:, 0]
    pitch_enu = euler_enu[:, 1]
    yaw_enu = euler_enu[:, 2]

    yaw_ned = (math.pi / 2) - yaw_enu
    yaw_ned = (yaw_ned + math.pi) % (2 * math.pi) - math.pi

    pitch_ned = -pitch_enu
    roll_ned = roll_enu

    return torch.stack([roll_ned, pitch_ned, yaw_ned], dim=1)