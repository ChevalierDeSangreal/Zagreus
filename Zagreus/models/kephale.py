import torch
import torch.nn as nn


class KephaleModuleVer0(nn.Module):
    """
    Based on TrackTransferModuleVer0
    """
    def __init__(self, device='cpu', embedding_size=64, traj_len=10, hidden_size=256, state_size=12):
        super(KephaleModuleVer0, self).__init__()
        self.device = device

        self.action_decoder = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4))
        self.traj_decoder = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_size * traj_len))
        self.world_model = None
        # Initialize Decision module
        self.decision_module = TrackAgileModuleVer2Dicision(input_size=9+3,device=device).to(device)


    

    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load the model's state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.eval()