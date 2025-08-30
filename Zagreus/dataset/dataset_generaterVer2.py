import numpy as np
import json
import os
from glob import glob
from pyulog import ULog
from scipy.spatial.transform import Rotation as R
from Zagreus.utils import ned_to_enu_xyz, ned_to_enu_euler
import torch

# -------------------------------
# 参数
# -------------------------------
ulog_dir = "/home/zim/Documents/Zagreus/Zagreus/data"
output_json = os.path.join(ulog_dir, "ulog_datasetVer2.json")
ulog_files = sorted(glob(os.path.join(ulog_dir, 'kephale_dataset', "*.ulg")))

def interp_array(t_src, data, t_target):
    return np.interp(t_target, t_src, data)

def interp_xyz(t_src, x, y, z, t_target):
    return (np.interp(t_target, t_src, x),
            np.interp(t_target, t_src, y),
            np.interp(t_target, t_src, z))

all_datasets = {}

for ulog_file in ulog_files:
    print(f"处理 {ulog_file} ...")
    ulog = ULog(ulog_file)

    # -------------------------------
    # 获取数据集
    # -------------------------------
    pos_data = ulog.get_dataset('vehicle_local_position').data
    att_data = ulog.get_dataset('vehicle_attitude_groundtruth').data
    ang_vel_data = ulog.get_dataset('vehicle_angular_velocity_groundtruth').data
    rates_setpoint = ulog.get_dataset('vehicle_rates_setpoint').data
    traj_sp = ulog.get_dataset('trajectory_setpoint').data

    # -------------------------------
    # 确定起始时间
    # -------------------------------
    traj_x = traj_sp['position[0]']
    traj_t = traj_sp['timestamp'] / 1e6  # 秒
    nonzero_idx = np.where(traj_x != 0)[0]
    if len(nonzero_idx) == 0:
        print(f"{ulog_file} 没有有效 trajectory_setpoint，跳过")
        continue
    t_start = traj_t[nonzero_idx[0]]
    print(f"起始时间: {t_start:.2f}s")

    # 结束时间
    t_pos = pos_data['timestamp'] / 1e6
    t_att = att_data['timestamp'] / 1e6
    t_vel = ang_vel_data['timestamp'] / 1e6
    t_ctrl = rates_setpoint['timestamp'] / 1e6
    t_end = min(t_pos[-1], t_att[-1], t_vel[-1], t_ctrl[-1])

    # 统一采样
    time_samples = np.arange(t_start, t_end, 0.02)  # 50Hz

    # -------------------------------
    # 插值 + 坐标转换
    # -------------------------------
    # 位置
    x_ned, y_ned, z_ned = interp_xyz(t_pos, pos_data['x'], pos_data['y'], pos_data['z'], time_samples)
    pos_enu = ned_to_enu_xyz(torch.tensor(np.stack([x_ned, y_ned, z_ned], axis=1), dtype=torch.float32))
    x, y, z = pos_enu[:,0].numpy(), pos_enu[:,1].numpy(), pos_enu[:,2].numpy()

    # 速度
    vx_ned, vy_ned, vz_ned = interp_xyz(t_pos, pos_data['vx'], pos_data['vy'], pos_data['vz'], time_samples)
    vel_enu = ned_to_enu_xyz(torch.tensor(np.stack([vx_ned, vy_ned, vz_ned], axis=1), dtype=torch.float32))
    vx, vy, vz = vel_enu[:,0].numpy(), vel_enu[:,1].numpy(), vel_enu[:,2].numpy()

    # 姿态 (四元数 -> 欧拉角)
    quats = np.stack([att_data['q[0]'], att_data['q[1]'], att_data['q[2]'], att_data['q[3]']], axis=1)
    rot_scipy = R.from_quat(np.stack([quats[:,1], quats[:,2], quats[:,3], quats[:,0]], axis=1))
    euler = rot_scipy.as_euler('xyz', degrees=False)
    roll_ned = np.interp(time_samples, t_att, euler[:,0])
    pitch_ned = np.interp(time_samples, t_att, euler[:,1])
    yaw_ned = np.interp(time_samples, t_att, euler[:,2])
    euler_enu = ned_to_enu_euler(torch.tensor(np.stack([roll_ned, pitch_ned, yaw_ned], axis=1), dtype=torch.float32))
    roll, pitch, yaw = euler_enu[:,0].numpy(), euler_enu[:,1].numpy(), euler_enu[:,2].numpy()

    # 角速度
    droll_ned = np.interp(time_samples, t_vel, ang_vel_data['xyz[0]'])
    dpitch_ned = np.interp(time_samples, t_vel, ang_vel_data['xyz[1]'])
    dyaw_ned = np.interp(time_samples, t_vel, ang_vel_data['xyz[2]'])
    angvel_enu = ned_to_enu_xyz(torch.tensor(np.stack([droll_ned, dpitch_ned, dyaw_ned], axis=1), dtype=torch.float32))
    droll, dpitch, dyaw = angvel_enu[:,0].numpy(), angvel_enu[:,1].numpy(), angvel_enu[:,2].numpy()

    # 控制指令
    droll_cmd = interp_array(t_ctrl, rates_setpoint['roll'], time_samples)
    dpitch_cmd = interp_array(t_ctrl, rates_setpoint['pitch'], time_samples)
    dyaw_cmd = interp_array(t_ctrl, rates_setpoint['yaw'], time_samples)
    thrust = interp_array(t_ctrl, rates_setpoint['thrust_body[2]'], time_samples)
    ctrl_cmds_ned = np.stack([droll_cmd, dpitch_cmd, dyaw_cmd], axis=1)
    ctrl_cmds_enu = ned_to_enu_xyz(torch.tensor(ctrl_cmds_ned, dtype=torch.float32)).numpy()
    droll_cmd, dpitch_cmd, dyaw_cmd = ctrl_cmds_enu.T
    thrust = -thrust  # ENU

    # -------------------------------
    # 存入总字典
    # -------------------------------
    dataset = {
        "timestamp": time_samples.tolist(),
        "x": x.tolist(), "y": y.tolist(), "z": z.tolist(),
        "vx": vx.tolist(), "vy": vy.tolist(), "vz": vz.tolist(),
        "roll": roll.tolist(), "pitch": pitch.tolist(), "yaw": yaw.tolist(),
        "droll": droll.tolist(), "dpitch": dpitch.tolist(), "dyaw": dyaw.tolist(),
        "droll_cmd": droll_cmd.tolist(), "dpitch_cmd": dpitch_cmd.tolist(), "dyaw_cmd": dyaw_cmd.tolist(),
        "thrust": thrust.tolist()
    }

    base = os.path.splitext(os.path.basename(ulog_file))[0]
    all_datasets[base] = dataset

with open(output_json, 'w') as f:
    json.dump(all_datasets, f, indent=2)

print(f"所有 ULog 数据已保存到 {output_json}")
