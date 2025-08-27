import json
import torch
from torch.utils.data import Dataset, DataLoader

class TrajDataset(Dataset):
    def __init__(self, json_path, seq_len=50, device="cpu"):
        with open(json_path, "r") as f:
            data = json.load(f)

        # 转换成 tensor
        self.device = device
        self.seq_len = seq_len

        self.x = torch.tensor(data["x"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.float32)
        self.z = torch.tensor(data["z"], dtype=torch.float32)
        self.vx = torch.tensor(data["vx"], dtype=torch.float32)
        self.vy = torch.tensor(data["vy"], dtype=torch.float32)
        self.vz = torch.tensor(data["vz"], dtype=torch.float32)
        self.roll = torch.tensor(data["roll"], dtype=torch.float32)
        self.pitch = torch.tensor(data["pitch"], dtype=torch.float32)
        self.yaw = torch.tensor(data["yaw"], dtype=torch.float32)
        self.droll = torch.tensor(data["droll"], dtype=torch.float32)
        self.dpitch = torch.tensor(data["dpitch"], dtype=torch.float32)
        self.dyaw = torch.tensor(data["dyaw"], dtype=torch.float32)

        self.droll_cmd = torch.tensor(data["droll_cmd"], dtype=torch.float32)
        self.dpitch_cmd = torch.tensor(data["dpitch_cmd"], dtype=torch.float32)
        self.dyaw_cmd = torch.tensor(data["dyaw_cmd"], dtype=torch.float32)
        self.thrust = torch.tensor(data["thrust"], dtype=torch.float32)

        # 拼接成状态和动作序列
        self.states = torch.stack([
            self.x, self.y, self.z,
            self.roll, self.pitch, self.yaw,
            self.vx, self.vy, self.vz,
            self.droll, self.dpitch, self.dyaw
        ], dim=1)

        self.actions = torch.stack([
            self.thrust, self.droll_cmd, self.dpitch_cmd, self.dyaw_cmd
        ], dim=1)

        self.len = len(self.states) - seq_len + 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        state_seq = self.states[idx:idx+self.seq_len].to(self.device)
        action_seq = self.actions[idx:idx+self.seq_len].to(self.device)
        return action_seq, state_seq


def get_dataloader(json_path, seq_len=50, batch_size=32, device="cpu", shuffle=True):
    dataset = TrajDataset(json_path, seq_len=seq_len, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
