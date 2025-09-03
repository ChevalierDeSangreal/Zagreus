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
from Zagreus.envs.base.dynamics_learnable import LearnableDynamics
from Zagreus.dataset import TrajDatasetVer2
from Zagreus.config import ROOT_DIR

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def get_time():
    timestamp = time.time()
    dt_object_utc = datetime.utcfromtimestamp(timestamp)
    target_timezone = pytz.timezone("Asia/Shanghai")
    dt_object_local = dt_object_utc.replace(tzinfo=pytz.utc).astimezone(target_timezone)
    formatted_time_local = dt_object_local.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_time_local

def get_args():
    parser = argparse.ArgumentParser(description="Trajectory Training Args")

    # ------------------- 基本任务参数 -------------------
    parser.add_argument("--experiment_name", type=str, default="test_dynamicsVer3", help="Name of the experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # ------------------- 环境 & 设备 -------------------
    parser.add_argument("--sim_device_type", type=str, default="cuda", choices=["cuda", "cpu"], help="Device type")
    parser.add_argument("--compute_device_id", type=int, default=0, help="CUDA device id")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments")

    # ------------------- 训练超参数 -------------------
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--num_epoch", type=int, default=500, help="Number of epochs")
    parser.add_argument("--seq_len", type=int, default=150, help="Sequence length for trajectory dataset")

    # ------------------- 数据路径 -------------------
    parser.add_argument("--data_name", type=str, default="ulog_datasetVer2.json", help="Trajectory dataset JSON file")
    parser.add_argument("--param_save_name", type=str, default="worldVer3.pth", help="Path to save model")
    parser.add_argument("--param_load_path", type=str, default="worldVer3.pth", help="Path to load pre-trained model")

    # ------------------- 其他 -------------------
    parser.add_argument("--num_worker", type=int, default=4, help="Number of workers for DataLoader")

    args = parser.parse_args()

    # ------------------- 设备组合 -------------------
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"


    return args
if __name__ == "__main__":
    args = get_args()
    run_name = f"{args.experiment_name}__{args.seed}__{get_time()}"

    param_save_path = os.path.join(ROOT_DIR, 'param', args.param_save_name)
    param_load_path = os.path.join(ROOT_DIR, 'param', args.param_load_path)
    data_path = os.path.join(ROOT_DIR, 'data', args.data_name)
    log_path = os.path.join(ROOT_DIR, 'test_runs', run_name)

    writer = SummaryWriter(log_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.sim_device
    print("Using device:", device)


    # Dynamics
    dynamics = LearnableDynamics(num_env=args.batch_size, dt=0.02, device=device).to(device)
    checkpoint = torch.load(param_load_path, map_location=device)
    dynamics.load_state_dict(checkpoint['model_state_dict'])
    dynamics.eval()

    # Dataset & Dataloader
    dataset = TrajDatasetVer2(data_path, seq_len=args.seq_len, device=device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Optimizer & Loss
    optimizer = optim.Adam(dynamics.parameters(), lr=args.learning_rate, eps=1e-5)
    criterion = nn.MSELoss()
    

    with torch.no_grad():
        for batch_idx, (action_seq, state_seq) in enumerate(loader):
            action_seq = action_seq.to(device)
            state_seq = state_seq.to(device)

            
            now_state = state_seq[:, 0].clone()  # 初始状态
            dynamics.reset_controller()  # 重置
            for step in range(args.seq_len):
                now_state, _ = dynamics(action_seq[:, step - 1], now_state)
                
                print("Step:", step, "State: ", now_state[0])
                writer.add_scalar(f'Position/X', now_state[0, 0], step)
                writer.add_scalar(f'Position/Y', now_state[0, 1], step)
                writer.add_scalar(f'Position/Z', now_state[0, 2], step)
                writer.add_scalar(f'True Position/X', state_seq[0, step, 0], step)
                writer.add_scalar(f'True Position/Y', state_seq[0, step, 1], step)
                writer.add_scalar(f'True Position/Z', state_seq[0, step, 2], step)
                writer.add_scalar(f'Velocity/X', now_state[0, 6], step)
                writer.add_scalar(f'Velocity/Y', now_state[0, 7], step)
                writer.add_scalar(f'Velocity/Z', now_state[0, 8], step)
                writer.add_scalar(f'Angular Velocity/X', now_state[0, 9], step)
                writer.add_scalar(f'Angular Velocity/Y', now_state[0, 10], step)
                writer.add_scalar(f'Angular Velocity/Z', now_state[0, 11], step)
                writer.add_scalar(f'True Velocity/X', state_seq[0, step, 6], step)
                writer.add_scalar(f'True Velocity/Y', state_seq[0, step, 7], step)
                writer.add_scalar(f'True Velocity/Z', state_seq[0, step, 8], step)
                writer.add_scalar(f'True Angular Velocity/X', state_seq[0, step, 9], step)
                writer.add_scalar(f'True Angular Velocity/Y', state_seq[0, step, 10], step)
                writer.add_scalar(f'True Angular Velocity/Z', state_seq[0, step, 11], step)
                writer.add_scalar(f'Euler Angle/Roll', now_state[0, 3], step)
                writer.add_scalar(f'Euler Angle/Pitch', now_state[0, 4], step)
                writer.add_scalar(f'Euler Angle/Yaw', now_state[0, 5], step)
                writer.add_scalar(f'True Euler Angle/Roll', state_seq[0, step, 3], step)
                writer.add_scalar(f'True Euler Angle/Pitch', state_seq[0, step, 4], step)
                writer.add_scalar(f'True Euler Angle/Yaw', state_seq[0, step, 5], step)
                writer.add_scalar(f'Thrust', action_seq[0, step - 1, 0], step)
                writer.add_scalar(f'Action/Roll Rate', action_seq[0, step - 1, 1], step)
                writer.add_scalar(f'Action/Pitch Rate', action_seq[0, step - 1, 2], step)
                writer.add_scalar(f'Action/Yaw Rate', action_seq[0, step - 1, 3], step)

            break

    writer.close()
    print("Testing complete!")
