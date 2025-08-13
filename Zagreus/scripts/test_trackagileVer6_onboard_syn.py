import os
import random
import time
import asyncio
from collections import deque

import isaacgym  # noqa
from isaacgym import gymutil
from isaacgym.torch_utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from pytorch3d.transforms import euler_angles_to_matrix

import pytz
from datetime import datetime
import sys

from Zagreus.config import ROOT_DIR
from Zagreus.utils import AgileLoss, agile_lossVer7
from Zagreus.utils import enu_to_ned_euler, ned_to_enu_euler, enu_to_ned_xyz, ned_to_enu_xyz
from Zagreus.models import TrackAgileModuleVer11
from Zagreus.envs import IrisDynamics, task_registry

from mavsdk import System
from mavsdk.offboard import (AttitudeRate, OffboardError)
from mavsdk.offboard import PositionNedYaw

# 全局变量用于数据共享
latest_odometry = None
latest_attitude = None
data_lock = asyncio.Lock()

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "track_agileVer3", "help": "The name of the task."},
        {"name": "--experiment_name", "type": str, "default": "test_trackagileVer6_onboard", "help": "Name of the experiment to run or load."},
        {"name": "--headless", "action": "store_true", "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--num_envs", "type": int, "default": 2, "help": "Number of environments to create. Batch size will be equal to this"},
        {"name": "--seed", "type": int, "default": 42, "help": "Random seed. Overrides config file if provided."},

        # train setting
        {"name": "--learning_rate", "type":float, "default": 1.6e-4,
            "help": "the learning rate of the optimizer"},
        {"name": "--batch_size", "type":int, "default": 2,
            "help": "batch size of training. Notice that batch_size should be equal to num_envs"},
        {"name": "--num_worker", "type":int, "default": 4,
            "help": "num worker of dataloader"},
        {"name": "--num_epoch", "type":int, "default": 500,
            "help": "num of epoch"},
        {"name": "--len_sample", "type":int, "default": 50,
            "help": "length of a sample"},
        {"name": "--tmp", "type": bool, "default": False, "help": "Set false to officially save the trainning log"},
        {"name": "--gamma", "type":int, "default": 0.8,
            "help": "how much will learning rate decrease"},
        {"name": "--slide_size", "type":int, "default": 10,
            "help": "size of GRU input window"},
        {"name": "--step_size", "type":int, "default": 100,
            "help": "learning rate will decrease every step_size steps"},

        # model setting
        {"name": "--param_save_name", "type":str, "default": 'track_agileVer6.pth',
            "help": "The path to model parameters"},
        {"name": "--param_load_path", "type":str, "default": 'track_agileVer6.pth',
            "help": "The path to model parameters"},
        
        ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="APG Policy",
        custom_parameters=custom_parameters)
    assert args.batch_size == args.num_envs, "batch_size should be equal to num_envs"

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"

    return args

def get_time():
    timestamp = time.time()
    dt_object_utc = datetime.utcfromtimestamp(timestamp)
    target_timezone = pytz.timezone("Asia/Shanghai")
    dt_object_local = dt_object_utc.replace(tzinfo=pytz.utc).astimezone(target_timezone)
    formatted_time_local = dt_object_local.strftime("%Y-%m-%d %H:%M:%S %Z")
    return formatted_time_local

class TimeProfiler:
    """时间性能分析器"""
    def __init__(self):
        self.times = {}
        self.counts = {}
    
    def record(self, name, duration):
        if name not in self.times:
            self.times[name] = []
            self.counts[name] = 0
        self.times[name].append(duration)
        self.counts[name] += 1
    
    def get_stats(self, name):
        if name not in self.times or not self.times[name]:
            return {"avg": 0, "max": 0, "min": 0, "count": 0}
        times = self.times[name]
        return {
            "avg": np.mean(times),
            "max": np.max(times),
            "min": np.min(times),
            "count": len(times),
            "latest": times[-1] if times else 0
        }

async def telemetry_updater(drone):
    """持续更新遥测数据的异步任务"""
    global latest_odometry, latest_attitude, data_lock
    
    async def update_odometry():
        async for odometry in drone.telemetry.odometry():
            async with data_lock:
                latest_odometry = odometry
    
    async def update_attitude():
        async for attitude in drone.telemetry.attitude_euler():
            async with data_lock:
                latest_attitude = attitude
    
    # 并行运行两个遥测更新任务
    await asyncio.gather(
        update_odometry(),
        update_attitude()
    )

async def get_current_state(device, args):
    """获取当前无人机状态"""
    global latest_odometry, latest_attitude, data_lock
    
    async with data_lock:
        if latest_odometry is None or latest_attitude is None:
            return None
        
        now_quad_state = torch.zeros((args.batch_size, 12), device=device)
        
        # 处理位置和速度数据
        ned_position = torch.tensor([
            latest_odometry.position_body.x_m, 
            latest_odometry.position_body.y_m, 
            latest_odometry.position_body.z_m
        ], device=device).unsqueeze(0)
        now_quad_state[:, :3] = ned_to_enu_xyz(ned_position)
        
        ned_velocity = torch.tensor([
            latest_odometry.velocity_body.x_m_s, 
            latest_odometry.velocity_body.y_m_s, 
            latest_odometry.velocity_body.z_m_s
        ], device=device).unsqueeze(0)
        now_quad_state[:, 6:9] = ned_to_enu_xyz(ned_velocity)
        
        # 处理角速度数据
        ned_angular_velocity = torch.tensor([
            latest_odometry.angular_velocity_body.roll_rad_s, 
            latest_odometry.angular_velocity_body.pitch_rad_s, 
            latest_odometry.angular_velocity_body.yaw_rad_s
        ], device=device).unsqueeze(0)
        now_quad_state[:, 9:] = ned_to_enu_euler(ned_angular_velocity)
        
        # 处理姿态数据
        ned_angle = torch.tensor([
            latest_attitude.roll_deg, 
            latest_attitude.pitch_deg, 
            latest_attitude.yaw_deg
        ], device=device).unsqueeze(0) / 180.0 * np.pi
        now_quad_state[:, 3:6] = ned_to_enu_euler(ned_angle)
        
        return now_quad_state

async def control_loop(drone, model, args, device, writer):
    """主控制循环"""
    profiler = TimeProfiler()
    
    # 初始化控制参数
    dt = 0.02  # 控制间隔
    mass = 1.9
    tar_acc_intervel = 100
    tar_acc_norm = 0
    theta = torch.rand(2, device=device) * 2 * torch.tensor(np.pi, device=device)
    x = tar_acc_norm * torch.cos(theta)
    y = tar_acc_norm * torch.sin(theta)
    tar_acc = torch.stack([x, y], dim=1).to(device)
    
    # 初始化模型相关变量
    tar_ori = torch.zeros((args.batch_size, 3)).to(device)
    init_vec = torch.tensor([[1.0, 0.0, 0.0]] * args.batch_size, device=device).unsqueeze(-1)
    old_loss = AgileLoss(args.batch_size, device=device)
    timer = torch.zeros((args.batch_size,), device=device)
    input_buffer = torch.zeros(args.slide_size, args.batch_size, 6+3).to(device)
    last_action = None
    
    # 等待初始数据
    print("等待初始遥测数据...")
    while True:
        now_quad_state = await get_current_state(device, args)
        if now_quad_state is not None:
            break
        await asyncio.sleep(0.001)
    
    tar_state = now_quad_state.clone().detach()
    tar_state[:, 0] += 2  # 目标位置向前2米
    
    print("开始控制循环...")
    
    # 主控制循环
    for step in range(args.len_sample):
        loop_start_time = time.time()
        
        # 获取当前状态
        state_start_time = time.time()
        now_quad_state = await get_current_state(device, args)
        if now_quad_state is None:
            print(f"警告: 步骤 {step} - 无法获取状态数据")
            continue
        state_time = time.time() - state_start_time
        profiler.record("state_acquisition", state_time)
        
        # 计算相对距离和变换
        calc_start_time = time.time()
        rel_dis = tar_state[:, :3] - now_quad_state[:, :3]
        world_to_body = IrisDynamics().world_to_body_matrix(now_quad_state[:, 3:6].detach())
        
        body_rel_dis = torch.matmul(world_to_body, torch.unsqueeze(rel_dis, 2)).squeeze(-1)
        body_vel = torch.matmul(world_to_body, torch.unsqueeze(now_quad_state[:, 6:9], 2)).squeeze(-1)
        
        tmp_input = torch.cat((body_vel, now_quad_state[:, 3:6], body_rel_dis), dim=1)
        tmp_input = tmp_input.unsqueeze(0)
        input_buffer = input_buffer[1:].clone()
        input_buffer = torch.cat((input_buffer, tmp_input), dim=0)
        calc_time = time.time() - calc_start_time
        profiler.record("calculation", calc_time)
        
        # 模型推理
        inference_start_time = time.time()
        with torch.no_grad():
            action = model.decision_module(input_buffer.clone())
        inference_time = time.time() - inference_start_time
        profiler.record("inference", inference_time)
        
        # 准备控制指令
        input_action = torch.zeros((args.batch_size, 4), device=device)
        input_action[:, 0] = action[:, 0] * 1
        input_action[:, 1:] = enu_to_ned_euler((action[:, 1:] * 2 - 1) * 3) * 180 / np.pi
        
        # 发送控制指令
        command_start_time = time.time()
        try:
            await drone.offboard.set_attitude_rate(AttitudeRate(
                input_action[0, 1], input_action[0, 2], input_action[0, 3], input_action[0, 0]
            ))
        except Exception as e:
            print(f"警告: 控制指令发送失败 - {e}")
        command_time = time.time() - command_start_time
        profiler.record("command", command_time)
        
        # 更新目标状态
        if (step % tar_acc_intervel) == 0:
            tar_acc *= -1
        if (step % (tar_acc_intervel * 2)) == 0:
            tar_acc *= -1
        tar_state[:, :2] = tar_state[:, :2] + tar_state[:, 6:8] * dt + tar_acc * 0.5 * dt * dt
        tar_state[:, 6:8] = tar_acc * dt + tar_state[:, 6:8]
        tar_pos = tar_state[:, :3].detach()
        
        # 计算损失和记录数据
        loss_start_time = time.time()
        loss_agile, new_loss = agile_lossVer7(
            old_loss, now_quad_state, tar_state, tar_state[:, 2].clone(), 
            tar_ori, 2, timer, dt, init_vec, action, last_action
        )
        old_loss = new_loss
        
        rotation_matrices = euler_angles_to_matrix(now_quad_state[:, 3:6], convention='XYZ')
        direction_vector = rotation_matrices @ init_vec
        direction_vector = direction_vector.squeeze()
        
        cos_sim = F.cosine_similarity(direction_vector, rel_dis, dim=1)
        theta = torch.acos(torch.clamp(cos_sim, -1, 1))
        theta_degrees = theta * 180.0 / np.pi
        
        cos_sim_hor = F.cosine_similarity(direction_vector[:, :2], rel_dis[:, :2], dim=1)
        theta_hor = torch.acos(torch.clamp(cos_sim_hor, -1, 1))
        theta_degrees_hor = theta_hor * 180.0 / np.pi
        loss_time = time.time() - loss_start_time
        profiler.record("loss_calculation", loss_time)
        
        # 记录日志
        item_tested = 0
        horizon_dis = torch.norm(now_quad_state[item_tested, :2] - tar_pos[item_tested, :2], dim=0, p=4)
        speed = torch.norm(now_quad_state[item_tested, 6:9], dim=0, p=2)
        
        # 记录性能指标
        profiler_stats = profiler.get_stats("inference")
        writer.add_scalar(f'Performance/Inference Time (ms)', profiler_stats["latest"] * 1000, step)
        writer.add_scalar(f'Performance/State Acquisition Time (ms)', profiler.get_stats("state_acquisition")["latest"] * 1000, step)
        writer.add_scalar(f'Performance/Calculation Time (ms)', profiler.get_stats("calculation")["latest"] * 1000, step)
        writer.add_scalar(f'Performance/Command Time (ms)', profiler.get_stats("command")["latest"] * 1000, step)
        
        # 记录控制指标
        writer.add_scalar(f'Total Loss', loss_agile[item_tested], step)
        writer.add_scalar(f'Direction Loss/sum', old_loss.direction[item_tested], step)
        writer.add_scalar(f'Direction Loss/xy', old_loss.direction_hor[item_tested], step)
        writer.add_scalar(f'Direction Loss/z', old_loss.direction_ver[item_tested], step)
        writer.add_scalar(f'Distance Loss', old_loss.distance[item_tested], step)
        writer.add_scalar(f'Velocity Loss', old_loss.vel[item_tested], step)
        writer.add_scalar(f'Orientation Loss', old_loss.ori[item_tested], step)
        writer.add_scalar(f'Smooth Loss', old_loss.aux[item_tested], step)
        writer.add_scalar(f'Height Loss', old_loss.h[item_tested], step)

        writer.add_scalar(f'Orientation/X', direction_vector[item_tested, 0], step)
        writer.add_scalar(f'Orientation/Y', direction_vector[item_tested, 1], step)
        writer.add_scalar(f'Orientation/Z', direction_vector[item_tested, 2], step)
        writer.add_scalar(f'Orientation/Theta', theta_degrees[item_tested], step)
        writer.add_scalar(f'Orientation/ThetaXY', theta_degrees_hor[item_tested], step)
        writer.add_scalar(f'Euler Angle/roll', now_quad_state[item_tested, 3], step)
        writer.add_scalar(f'Euler Angle/pitch', now_quad_state[item_tested, 4], step)
        writer.add_scalar(f'Euler Angle/yaw', now_quad_state[item_tested, 5], step)
        writer.add_scalar(f'Angular Velocity/roll', now_quad_state[item_tested, 9], step)
        writer.add_scalar(f'Angular Velocity/pitch', now_quad_state[item_tested, 10], step)
        writer.add_scalar(f'Angular Velocity/yaw', now_quad_state[item_tested, 11], step)
        writer.add_scalar(f'Horizon Distance', horizon_dis, step)
        writer.add_scalar(f'Position/X', now_quad_state[item_tested, 0], step)
        writer.add_scalar(f'Position/Y', now_quad_state[item_tested, 1], step)
        writer.add_scalar(f'Target Position/X', tar_pos[item_tested, 0], step)
        writer.add_scalar(f'Target Position/Y', tar_pos[item_tested, 1], step)
        writer.add_scalar(f'Velocity/X', now_quad_state[item_tested, 6], step)
        writer.add_scalar(f'Velocity/Y', now_quad_state[item_tested, 7], step)
        writer.add_scalar(f'Velocity/Z', now_quad_state[item_tested, 8], step)
        writer.add_scalar(f'Distance/X', tar_pos[item_tested, 0] - now_quad_state[item_tested, 0], step)
        writer.add_scalar(f'Distance/Y', tar_pos[item_tested, 1] - now_quad_state[item_tested, 1], step)
        writer.add_scalar(f'Speed/Z', now_quad_state[item_tested, 8], step)
        writer.add_scalar(f'Speed', speed, step)
        writer.add_scalar(f'Height', now_quad_state[item_tested, 2], step)

        writer.add_scalar(f'Action/F', input_action[item_tested, 0] * np.pi / 180, step)
        writer.add_scalar(f'Action/roll', input_action[item_tested, 1] * np.pi / 180, step)
        writer.add_scalar(f'Action/pitch', input_action[item_tested, 2] * np.pi / 180, step)
        writer.add_scalar(f'Action/yaw', input_action[item_tested, 3] * np.pi / 180, step)
        
        timer = timer + 1
        last_action = action.clone().detach()
        
        # 控制循环时间
        loop_total_time = time.time() - loop_start_time
        profiler.record("total_loop", loop_total_time)
        writer.add_scalar(f'Performance/Total Loop Time (ms)', loop_total_time * 1000, step)
        
        # 打印调试信息
        if step % 10 == 0 or step < 5:  # 每10步或前5步打印一次
            inference_stats = profiler.get_stats("inference")
            total_stats = profiler.get_stats("total_loop")
            print(f"步骤 {step}: 推理时间 {inference_stats['latest']*1000:.1f}ms, "
                  f"总循环时间 {total_stats['latest']*1000:.1f}ms")
            print(f"位置: [{now_quad_state[0, 0]:.2f}, {now_quad_state[0, 1]:.2f}, {now_quad_state[0, 2]:.2f}]")
            print(f"目标: [{tar_pos[0, 0]:.2f}, {tar_pos[0, 1]:.2f}, {tar_pos[0, 2]:.2f}]")
            print(f"动作: [F={input_action[0, 0]:.2f}, R={input_action[0, 1]:.2f}, P={input_action[0, 2]:.2f}, Y={input_action[0, 3]:.2f}]")
            print("-" * 60)
        
        # 精确控制时间间隔
        remaining_time = dt - loop_total_time
        if remaining_time > 0:
            await asyncio.sleep(remaining_time)
        else:
            print(f"警告: 步骤 {step} 超时 {abs(remaining_time)*1000:.1f}ms")
    
    # 打印最终统计
    print("\n=== 性能统计 ===")
    for name in ["inference", "state_acquisition", "calculation", "command", "total_loop"]:
        stats = profiler.get_stats(name)
        print(f"{name}: 平均 {stats['avg']*1000:.2f}ms, "
              f"最大 {stats['max']*1000:.2f}ms, "
              f"最小 {stats['min']*1000:.2f}ms, "
              f"次数 {stats['count']}")

async def run():
    """主运行函数"""
    args = get_args()
    run_name = f"{args.task}__{args.experiment_name}__{args.seed}__{get_time()}"
    if args.tmp:
        run_name = 'tmp_' + run_name
    
    param_load_path = os.path.join(ROOT_DIR, 'param', args.param_load_path)
    log_path = os.path.join(ROOT_DIR, 'test_runs', run_name)

    writer = SummaryWriter(log_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = args.sim_device
    print("使用设备:", device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 初始化模型
    model = TrackAgileModuleVer11(device=device).to(device)
    model.load_model(param_load_path)
    model.eval()
    
    # 连接无人机
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("等待无人机连接...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- 已连接到无人机!")
            break

    print("-- 解锁")
    await drone.action.arm()

    print("-- 起飞")
    await drone.action.takeoff()
    await asyncio.sleep(10)

    # 获取初始高度并设置悬停
    async for odometry in drone.telemetry.odometry():
        desired_z = odometry.position_body.z_m
        break

    target_yaw_deg = 0.0
    await drone.offboard.set_position_ned(PositionNedYaw(north_m=0, east_m=0, down_m=desired_z, yaw_deg=target_yaw_deg))
    await drone.offboard.start()

    print("-- 初始化位置")
    for _ in range(40):
        await drone.offboard.set_position_ned(PositionNedYaw(north_m=0, east_m=0, down_m=desired_z, yaw_deg=target_yaw_deg))
        await asyncio.sleep(0.05)

    print("开始并行任务...")
    
    # 启动遥测数据更新和控制循环
    try:
        await asyncio.gather(
            telemetry_updater(drone),
            control_loop(drone, model, args, device, writer)
        )
    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"运行出错: {e}")
    finally:
        # 清理资源
        try:
            await drone.offboard.stop()
            await drone.action.land()
        except:
            pass
        writer.close()
        print("程序结束")

if __name__ == "__main__":
    asyncio.run(run())