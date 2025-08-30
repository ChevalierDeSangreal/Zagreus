import json
import torch
from torch.utils.data import Dataset, DataLoader

class TrajDatasetVer2(Dataset):
    def __init__(self, json_path, seq_len=50, device="cpu"):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.device = device
        self.seq_len = seq_len
        self.samples = []  # 存放每个 ulog 的数据

        for ulog_name, traj in data.items():
            # 转 tensor
            x = torch.tensor(traj["x"], dtype=torch.float32)
            y = torch.tensor(traj["y"], dtype=torch.float32)
            z = torch.tensor(traj["z"], dtype=torch.float32)
            vx = torch.tensor(traj["vx"], dtype=torch.float32)
            vy = torch.tensor(traj["vy"], dtype=torch.float32)
            vz = torch.tensor(traj["vz"], dtype=torch.float32)
            roll = torch.tensor(traj["roll"], dtype=torch.float32)
            pitch = torch.tensor(traj["pitch"], dtype=torch.float32)
            yaw = torch.tensor(traj["yaw"], dtype=torch.float32)
            droll = torch.tensor(traj["droll"], dtype=torch.float32)
            dpitch = torch.tensor(traj["dpitch"], dtype=torch.float32)
            dyaw = torch.tensor(traj["dyaw"], dtype=torch.float32)

            droll_cmd = torch.tensor(traj["droll_cmd"], dtype=torch.float32)
            dpitch_cmd = torch.tensor(traj["dpitch_cmd"], dtype=torch.float32)
            dyaw_cmd = torch.tensor(traj["dyaw_cmd"], dtype=torch.float32)
            thrust = torch.tensor(traj["thrust"], dtype=torch.float32)

            # 拼接状态和动作
            states = torch.stack([
                x, y, z,
                roll, pitch, yaw,
                vx, vy, vz,
                droll, dpitch, dyaw
            ], dim=1)

            actions = torch.stack([
                thrust, droll_cmd, dpitch_cmd, dyaw_cmd
            ], dim=1)

            # 如果轨迹长度不足 seq_len，就跳过
            if len(states) >= seq_len:
                self.samples.append((states, actions))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        states, actions = self.samples[idx]
        # 随机截取一个子序列（保证长度为 seq_len）
        start = torch.randint(0, len(states) - self.seq_len + 1, (1,)).item()
        state_seq = states[start:start+self.seq_len].to(self.device)
        action_seq = actions[start:start+self.seq_len].to(self.device)
        return action_seq, state_seq


def get_dataloader(json_path, seq_len=50, batch_size=32, device="cpu", shuffle=True):
    dataset = TrajDatasetVer2(json_path, seq_len=seq_len, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
