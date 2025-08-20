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
    parser.add_argument("--experiment_name", type=str, default="train_dynamicsVer0", help="Name of the experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # ------------------- 环境 & 设备 -------------------
    parser.add_argument("--sim_device_type", type=str, default="cuda", choices=["cuda", "cpu"], help="Device type")
    parser.add_argument("--compute_device_id", type=int, default=0, help="CUDA device id")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments")

    # ------------------- 训练超参数 -------------------
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--num_epoch", type=int, default=500, help="Number of epochs")
    parser.add_argument("--seq_len", type=int, default=100, help="Sequence length for trajectory dataset")
    parser.add_argument("--save_interval", type=int, default=50, help="Save model every N epochs")

    # ------------------- 数据路径 -------------------
    parser.add_argument("--data_name", type=str, default="ulog_dataset.json", help="Trajectory dataset JSON file")
    parser.add_argument("--param_save_name", type=str, default="worldVere0.pth", help="Path to save model")
    parser.add_argument("--param_load_path", type=str, default="", help="Path to load pre-trained model")

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
    log_path = os.path.join(ROOT_DIR, 'runs', run_name)

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
    dynamics = IrisDynamics(device=device).to(device)
    

    # Dataset & Dataloader
    dataset = TrajDataset(data_path, seq_len=args.seq_len, device=device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer & Loss
    optimizer = optim.Adam(dynamics.parameters(), lr=args.learning_rate, eps=1e-5)
    criterion = nn.MSELoss()

    # 训练循环
    for epoch in range(args.num_epoch):
        total_loss = 0.0
        for batch_idx, (action_seq, state_seq) in enumerate(loader):
            action_seq = action_seq.to(device)
            state_seq = state_seq.to(device)
            loss = torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            # print("Shape of state_seq:", state_seq[:, 0].shape)
            
            dynamics.reset_action()  # 重置动作
            for step in range(1, args.seq_len//2):
                now_state, _ = dynamics(action_seq[:, step - 1], state_seq[:, step - 1], dt=0.02)

            now_state = state_seq[:, args.seq_len//2].clone()
            for step in range(args.seq_len//2, args.seq_len):
                now_state, _ = dynamics(action_seq[:, step - 1], now_state, dt=0.02)
                loss = loss + criterion(now_state, state_seq[:, step])
            
            loss = loss / (args.seq_len / 2 - 1)  # 平均损失

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{args.num_epoch}], Loss: {avg_loss:.6f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # 每 N epoch 保存模型
        if (epoch + 1) % args.save_interval == 0:
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': dynamics.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }
            torch.save(save_dict, param_save_path)

    writer.close()
    print("Training complete!")
