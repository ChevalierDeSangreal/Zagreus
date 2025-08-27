import numpy as np
import json
from pyulog import ULog
from scipy.spatial.transform import Rotation as R
from Zagreus.utils import ned_to_enu_xyz, ned_to_enu_euler
import torch
# -------------------------------
# 1. 读取 ULog 文件
# -------------------------------
ulog_file = '/home/core/wangzimo/Zagreus/Zagreus/data/06_46_55.ulg'
ulog = ULog(ulog_file)

# 获取所需数据集
vehicle_local_position = ulog.get_dataset('vehicle_local_position')
vehicle_attitude_groundtruth = ulog.get_dataset('vehicle_attitude_groundtruth')
vehicle_angular_velocity_groundtruth = ulog.get_dataset('vehicle_angular_velocity_groundtruth')
vehicle_rates_setpoint = ulog.get_dataset('vehicle_rates_setpoint')

# -------------------------------
# 2. 提取原始数据
# -------------------------------
pos_data = vehicle_local_position.data
att_data = vehicle_attitude_groundtruth.data
ang_vel_data = vehicle_angular_velocity_groundtruth.data
rates_setpoint = vehicle_rates_setpoint.data

# -------------------------------
# 3. 构建统一时间轴
# -------------------------------
# 转换 timestamp 单位为秒
t_pos = pos_data['timestamp'] / 1e6
t_att = att_data['timestamp'] / 1e6
t_vel = ang_vel_data['timestamp'] / 1e6
t_ctrl = rates_setpoint['timestamp'] / 1e6

# 采样起始时间 (10s后)
t_start = 10.0
t_end = min(t_pos[-1], t_att[-1], t_vel[-1], t_ctrl[-1])

# 目标时间轴，间隔 0.02s
time_samples = np.arange(t_start, t_end, 0.02)

# -------------------------------
# 4. 使用线性插值获取各项数据
# -------------------------------
def interp_array(t_src, data, t_target):
    """按时间插值"""
    return np.interp(t_target, t_src, data)

def interp_xyz(t_src, x, y, z, t_target):
    return (np.interp(t_target, t_src, x),
            np.interp(t_target, t_src, y),
            np.interp(t_target, t_src, z))

# 位置
x_ned, y_ned, z_ned = interp_xyz(t_pos, pos_data['x'], pos_data['y'], pos_data['z'], time_samples)
pos_enu = ned_to_enu_xyz(torch.tensor(np.stack([x_ned, y_ned, z_ned], axis=1), dtype=torch.float32))
x, y, z = pos_enu[:,0].numpy(), pos_enu[:,1].numpy(), pos_enu[:,2].numpy()

# 速度
vx_ned, vy_ned, vz_ned = interp_xyz(t_pos, pos_data['vx'], pos_data['vy'], pos_data['vz'], time_samples)
vel_enu = ned_to_enu_xyz(torch.tensor(np.stack([vx_ned, vy_ned, vz_ned], axis=1), dtype=torch.float32))
vx, vy, vz = vel_enu[:,0].numpy(), vel_enu[:,1].numpy(), vel_enu[:,2].numpy()

# -------------------------------
# 5. 四元数转欧拉角
# -------------------------------
quats = np.stack([att_data['q[0]'], att_data['q[1]'], att_data['q[2]'], att_data['q[3]']], axis=1)
rot_scipy = R.from_quat(np.stack([quats[:,1], quats[:,2], quats[:,3], quats[:,0]], axis=1))
euler = rot_scipy.as_euler('xyz', degrees=False)  # roll, pitch, yaw

# 对 roll/pitch/yaw 进行时间插值
roll_ned = np.interp(time_samples, t_att, euler[:,0])
pitch_ned = np.interp(time_samples, t_att, euler[:,1])
yaw_ned = np.interp(time_samples, t_att, euler[:,2])

# NED转ENU
euler_enu = ned_to_enu_euler(torch.tensor(np.stack([roll_ned, pitch_ned, yaw_ned], axis=1), dtype=torch.float32))
roll, pitch, yaw = euler_enu[:,0].numpy(), euler_enu[:,1].numpy(), euler_enu[:,2].numpy()

# 当前角速度
droll_ned = np.interp(time_samples, t_vel, ang_vel_data['xyz[0]'])
dpitch_ned = np.interp(time_samples, t_vel, ang_vel_data['xyz[1]'])
dyaw_ned = np.interp(time_samples, t_vel, ang_vel_data['xyz[2]'])

angvel_enu = ned_to_enu_xyz(torch.tensor(np.stack([droll_ned, dpitch_ned, dyaw_ned], axis=1), dtype=torch.float32))
droll, dpitch, dyaw = angvel_enu[:,0].numpy(), angvel_enu[:,1].numpy(), angvel_enu[:,2].numpy()

# 当前角速度控制指令
droll_cmd = interp_array(t_ctrl, rates_setpoint['roll'], time_samples)
dpitch_cmd = interp_array(t_ctrl, rates_setpoint['pitch'], time_samples)
dyaw_cmd = interp_array(t_ctrl, rates_setpoint['yaw'], time_samples)
thrust = interp_array(t_ctrl, rates_setpoint['thrust_body[2]'], time_samples)

ctrl_cmds_ned = np.stack([droll_cmd, dpitch_cmd, dyaw_cmd], axis=1)
ctrl_cmds_enu = ned_to_enu_xyz(torch.tensor(ctrl_cmds_ned, dtype=torch.float32)).numpy()
droll_cmd, dpitch_cmd, dyaw_cmd = ctrl_cmds_enu.T  # 拆成时间序列

# 推力 (z轴取反)
thrust = -thrust

# -------------------------------
# 6. 构建字典并保存为 JSON
# -------------------------------
dataset = {
    "timestamp": time_samples,
    "x": x, "y": y, "z": z,
    "vx": vx, "vy": vy, "vz": vz,
    "roll": roll, "pitch": pitch, "yaw": yaw,
    "droll": droll, "dpitch": dpitch, "dyaw": dyaw,
    "droll_cmd": droll_cmd, "dpitch_cmd": dpitch_cmd, "dyaw_cmd": dyaw_cmd,
    "thrust": thrust
}

# 将 numpy array 转换为列表，以便 JSON 序列化
dataset_json = {k: v.tolist() for k, v in dataset.items()}
dataset_path = "/home/core/wangzimo/Zagreus/Zagreus/data/ulog_dataset.json"
# 保存
with open(dataset_path, 'w') as f:
    json.dump(dataset_json, f, indent=2)

print(f"数据集已保存到 {dataset_path}")
