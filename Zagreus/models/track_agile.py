import torch
import torch.nn as nn
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from Zagreus.models.track_transfer import TrackTransferModuleVer0Predict

class TrackAgileModuleVer0(nn.Module):
    """
    First Version of Tracking Agile Target
    """
    def __init__(self, input_size=6, hidden_size1=32, hidden_size2=32, output_size=3, device='cpu'):
        print("TrackSpaceModel Initializing...")

        super(TrackAgileModuleVer0, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ELU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ELU().to(device)
        self.output_layer = nn.Linear(hidden_size2, output_size).to(device)

        torch.nn.init.kaiming_normal_(self.hidden_layer1.weight)
        torch.nn.init.kaiming_normal_(self.hidden_layer2.weight)
        torch.nn.init.kaiming_normal_(self.output_layer.weight)


    def forward(self, now_state, rel_dis):
        
        x = torch.cat((now_state, rel_dis), dim=1)
        x = self.hidden_layer1(x)
        x = self.activation1(x)
        x = self.hidden_layer2(x)
        x = self.activation2(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x) * 2 - 1
        return x


class TrackAgileModuleVer1(nn.Module):
    def __init__(self, input_size=9+3, hidden_size=256, output_size=4, num_layers=2, device='cpu'):
        super(TrackAgileModuleVer1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        torch.nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size).to(x[0].device)
        # print(x.shape, h0.shape)
        out, _ = self.gru(x, h0)
        # print(out.shape)
        out = self.fc(out[-1, :, :])
        # print(out.shape)
        out = torch.sigmoid(out) * 2 - 1
        return out

class TrackAgileModuleVer2Dicision(nn.Module):
    def __init__(self, input_size=9+9, hidden_size=256, output_size=4, num_layers=2, device='cpu'):
        super(TrackAgileModuleVer2Dicision, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        torch.nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size).to(x[0].device)
        # print(x.shape, h0.shape)
        out, _ = self.gru(x, h0)
        # print(out.shape)
        out = self.fc(out[-1, :, :])
        # print(out.shape)
        out = torch.sigmoid(out) * 2 - 1
        return out
    
class TrackAgileModuleVer2ExtractorVer2(nn.Module):
    def __init__(self, device):
        super(TrackAgileModuleVer2ExtractorVer2, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=11, stride=11, padding=2)

        self.fc = nn.Sequential(
            nn.Linear(20 * 20, 80),
            nn.ReLU(),
            nn.Linear(80, 9)
        )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')  # For ReLU activations
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero (optional)
        self.device = device

    def forward(self, x, mask):
        x = torch.where(mask, x, torch.full_like(x, 333))
        x = -self.maxpooling(-x)
        x[x == 333] = 0
        # file_path = "/home/wangzimo/VTT/VTT/aerial_gym/scripts/camera_output/test_input.png"
        # image_to_visualize = x[0].cpu().numpy()
        # # print(x[0])
        # plt.figure(figsize=(6, 6))
        # plt.imshow(image_to_visualize, cmap='viridis', vmin=0, vmax=10)  # 可以根据需要更改 colormap
        # plt.colorbar()  # 添加颜色条以显示值范围
        # plt.title(f"Visualizing Image Input: Batch {0}")
        # plt.xlabel("X-axis")
        # plt.ylabel("Y-axis")
        # plt.savefig(file_path)
        # plt.close()
        # exit(0)
        # # print(mask[0])
        # # print(dep_image[0])
        # # print(image_input[0])

        # print(x.shape)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        # print(out.shape)
        # out = torch.sigmoid(out) * 2 - 1
        return out
    
class TrackAgileModuleVer2Extractor(nn.Module):
    def __init__(self, device):
        super(TrackAgileModuleVer2Extractor, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=5, stride=5, padding=0)

        self.fc = nn.Sequential(
            nn.Linear(44 * 44, 44),
            nn.ReLU(),
            nn.Linear(44, 9)
        )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')  # For ReLU activations
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero (optional)
        self.device = device

    def forward(self, x):
        
        x = -self.maxpooling(-x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        # print(out.shape)
        # out = torch.sigmoid(out) * 2 - 1
        return out
    
class DirectionPrediction(nn.Module):
    def __init__(self, device):
        super(DirectionPrediction, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 6),
            nn.ReLU(),
            nn.Linear(6, 3)
        )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')  # For ReLU activations
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero (optional)
        self.device = device
    def forward(self, x):
        x = self.fc(x)
        return x

class TrackAgileModuleVer3(nn.Module):
    def __init__(self, device='cpu'):
        super(TrackAgileModuleVer3, self).__init__()
        self.device = device

        # Initialize Decision module
        self.decision_module = TrackAgileModuleVer2Dicision(device=device).to(device)

        # Initialize Extractor module
        self.extractor_module = TrackAgileModuleVer2ExtractorVer2(device=device).to(device)

        self.directpred = DirectionPrediction(device=device).to(device)

    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load the model's state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.eval()

class TrackAgileModuleVer4(nn.Module):
    """
    Based on Ver3
    Use distance to output action
    """
    def __init__(self, device='cpu'):
        super(TrackAgileModuleVer4, self).__init__()
        self.device = device

        # Initialize Decision module
        self.decision_module = TrackAgileModuleVer2Dicision(input_size=9+3,device=device).to(device)

        # Initialize Extractor module
        self.extractor_module = TrackAgileModuleVer2ExtractorVer2(device=device).to(device)

        self.directpred = DirectionPrediction(device=device).to(device)

    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load the model's state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.eval()

class TrackAgileModuleVer5(nn.Module):
    """
    Based on Ver4
    No velocity to output action
    """
    def __init__(self, device='cpu'):
        super(TrackAgileModuleVer5, self).__init__()
        self.device = device

        # Initialize Decision module
        self.decision_module = TrackAgileModuleVer2Dicision(input_size=6+3,device=device).to(device)

        # Initialize Extractor module
        self.extractor_module = TrackAgileModuleVer2ExtractorVer2(device=device).to(device)

        self.directpred = DirectionPrediction(device=device).to(device)

    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load the model's state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.eval()

class TrackAgileModuleVer6(nn.Module):
    """
    Based on Ver5
    No velocity and attitude to output action
    """
    def __init__(self, device='cpu'):
        super(TrackAgileModuleVer6, self).__init__()
        self.device = device

        # Initialize Decision module
        self.decision_module = TrackAgileModuleVer2Dicision(input_size=3+3,device=device).to(device)

        # Initialize Extractor module
        self.extractor_module = TrackAgileModuleVer2ExtractorVer2(device=device).to(device)

        self.directpred = DirectionPrediction(device=device).to(device)

    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load the model's state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.eval()

class TrackAgileModuleVer3Dicision(nn.Module):
    def __init__(self, input_size=9+9, hidden_size=256, output_size=4, num_layers=2, device='cpu'):
        super(TrackAgileModuleVer3Dicision, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        torch.nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x, h0):
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size).to(x[0].device)

        # print(x.shape, h0.shape)
        out, hn = self.gru(x, h0)
        # print(out.shape)
        out = self.fc(out[-1, :, :])
        # print(out.shape)
        out = torch.sigmoid(out) * 2 - 1
        return out, hn
    
class TrackAgileModuleVer7(nn.Module):
    """
    Based on Ver6
    New GRU structure
    """
    def __init__(self, device='cpu'):
        super(TrackAgileModuleVer7, self).__init__()
        self.device = device

        # Initialize Decision module
        self.decision_module = TrackAgileModuleVer3Dicision(input_size=9+3,device=device).to(device)

        # Initialize Extractor module
        self.extractor_module = TrackAgileModuleVer2ExtractorVer2(device=device).to(device)

        self.directpred = DirectionPrediction(device=device).to(device)

    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load the model's state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.eval()

class TrackAgileModuleVer8(nn.Module):
    """
    Based on Ver6
    Add f to output action
    """
    def __init__(self, device='cpu'):
        super(TrackAgileModuleVer8, self).__init__()
        self.device = device

        # Initialize Decision module
        self.decision_module = TrackAgileModuleVer2Dicision(input_size=3+3+1,device=device).to(device)

        # Initialize Extractor module
        self.extractor_module = TrackAgileModuleVer2ExtractorVer2(device=device).to(device)

        self.directpred = DirectionPrediction(device=device).to(device)

    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load the model's state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    def set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.eval()

class TrackAgileModuleVer4Dicision(nn.Module):
    def __init__(self, input_size=9+9, hidden_size=256, output_size=4, num_layers=2, seq_len=10, device='cpu'):
        super(TrackAgileModuleVer4Dicision, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.gru = nn.GRU(input_size, hidden_size, num_layers).to(device)
        self.fc = nn.Linear(hidden_size, seq_len * output_size).to(device)
        torch.nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size).to(x[0].device)
        embedding, _ = self.gru(x, h0)
        out = self.fc(embedding[-1, :, :])
        out = out.view(x.shape[1], self.seq_len, -1)
        out = torch.sigmoid(out)
        return out, embedding[-1, :, :]
    

class TrackAgileModuleVer9(nn.Module):
    """
    Based on ver2
    Action output will be in range [0, 1]
    Do not use av as input
    """
    def __init__(self, device='cpu'):
        super(TrackAgileModuleVer9, self).__init__()
        self.device = device

        # Initialize Decision module
        self.decision_module = TrackAgileModuleVer4Dicision(input_size=6+3,device=device).to(device)

        self.predict_module = TrackTransferModuleVer0Predict(device=device).to(device)

    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load the model's state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    def set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.eval()

class TrackAgileModuleVer10Extractor(nn.Module):
    def __init__(self,
                 in_channels_prev=2,
                 in_channels_curr=1,
                 hidden_dim=16*16,     # token 特征维度，现在等于 16x16
                 cnn_feats=8,       # token 数量
                 num_heads=8,
                 mlp_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cnn_feats = cnn_feats

        # CNN for previous inputs (depth + seg)
        self.backbone_prev = nn.Sequential(
            nn.Conv2d(in_channels_prev, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, cnn_feats, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )  # -> (B, cnn_feats, 56, 56)
        self.pool_prev = nn.AdaptiveAvgPool2d((16, 16))  # -> (B, cnn_feats, 16, 16)

        # CNN for current depth
        self.backbone_curr = nn.Sequential(
            nn.Conv2d(in_channels_curr, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, cnn_feats, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool_curr = nn.AdaptiveAvgPool2d((16, 16))

        # Attention modules
        self.encoder_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                  num_heads=num_heads,
                                                  batch_first=True)
        self.decoder_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                  num_heads=num_heads,
                                                  batch_first=True)

        # LayerNorm for token embedding (optional)
        self.ln_tokens = nn.LayerNorm(hidden_dim)

        # MLP head
        self.mlp_out = nn.Sequential(
            nn.Linear(cnn_feats * hidden_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, mlp_dim)
        )

        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, 3)
        )

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Linear(mlp_dim, cnn_feats * 16 * 16),
            nn.ReLU(inplace=True)
        )
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(cnn_feats, 16, kernel_size=4, stride=2, padding=1),  # (16,16)→(32,32)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),         # (32,32)→(64,64)
            nn.Upsample(size=(223, 223), mode='bilinear', align_corners=False)     # (64,64)→(223,223)
        )

    def forward(self, d_prev, s_prev, d_curr):
        B = d_prev.shape[0]
        d_prev = d_prev.unsqueeze(1)
        s_prev = s_prev.unsqueeze(1)
        d_curr = d_curr.unsqueeze(1)

        ### ---- Encoder path ----
        x_prev = torch.cat([d_prev, s_prev], dim=1)  # (B,2,223,223)
        feat_prev = self.backbone_prev(x_prev)       # (B,32,56,56)
        feat_prev = self.pool_prev(feat_prev)        # (B,8,16,16)

        # Flatten spatial dims: (B, C, H, W) → (B, C, H*W) → tokens
        enc_tokens = feat_prev.view(B, self.cnn_feats, -1)  # (B, 8, 256)
        enc_tokens = self.ln_tokens(enc_tokens)             # (B, 8, 256)

        # Self-attention on encoder tokens
        enc_out, _ = self.encoder_attn(enc_tokens, enc_tokens, enc_tokens)  # (B, 8, 256)

        ### ---- Decoder path ----
        feat_curr = self.backbone_curr(d_curr)              # (B,8,56,56)
        feat_curr = self.pool_curr(feat_curr)               # (B,8,16,16)
        dec_tokens = feat_curr.view(B, self.cnn_feats, -1)  # (B,8,256)
        dec_tokens = self.ln_tokens(dec_tokens)             # (B,8,256)

        # Cross-attention: decoder tokens attend to encoder outputs
        dec_out, _ = self.decoder_attn(dec_tokens, enc_out, enc_out)  # (B, 8, 256)

        # print(dec_out.shape)
        ### ---- Output path ----
        dec_out_flat = dec_out.reshape(B, -1)                  # (B, 8*256)
        output_feature = self.mlp_out(dec_out_flat)         # (B, mlp_dim)

        rel_dist = self.regressor(output_feature)           # (B,3)

        seg_feat = self.seg_head(output_feature)            # (B, cnn_feats*16*16)
        
        seg_feat = seg_feat.view(B, self.cnn_feats, 16, 16) # (B,8,16,16)
        s_t = self.up_conv(seg_feat).squeeze(1)             # (B,223,223) 未激活的 logits
        
        s_t = s_t.squeeze(1)
        probs = torch.sigmoid(s_t)                        # (B,223,223) [0,1] 概率

        return {
            'rel_dist': rel_dist, # (B, 3)
            'segmentation': s_t, # (B, 223, 223)
            'feature': output_feature, # (B, mlp_dim)
            'segmentation_probs': probs  # (B, 223, 223) 概率
        }

class TrackAgileModuleVer10(nn.Module):
    """
    Based on ver9
    Added extractor module
    """
    def __init__(self, device='cpu'):
        super(TrackAgileModuleVer10, self).__init__()
        self.device = device

        image_feature_size = 64

        # Initialize Decision module
        self.decision_module = TrackAgileModuleVer4Dicision(input_size=6+image_feature_size,device=device).to(device)

        self.predict_module = TrackTransferModuleVer0Predict(device=device).to(device)

        self.extractor_module = TrackAgileModuleVer10Extractor(mlp_dim=image_feature_size).to(device)

    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load the model's state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    def set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.eval()


class TrackAgileModuleVer5Dicision(nn.Module):
    def __init__(self, input_size=9+9, hidden_size=256, output_size=4, num_layers=2, device='cpu'):
        super(TrackAgileModuleVer5Dicision, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        torch.nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size).to(x[0].device)
        # print(x.shape, h0.shape)
        out, _ = self.gru(x, h0)
        # print(out.shape)
        out = self.fc(out[-1, :, :])
        # print(out.shape)
        out = torch.sigmoid(out)
        return out
    
class TrackAgileModuleVer11(nn.Module):
    """
    Based on ver10
    Do not use embedding framework
    """
    def __init__(self, device='cpu'):
        super(TrackAgileModuleVer11, self).__init__()
        self.device = device


        # Initialize Decision module
        self.decision_module = TrackAgileModuleVer5Dicision(input_size=6+3,device=device).to(device)

        self.predict_module = TrackTransferModuleVer0Predict(device=device).to(device)

    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load the model's state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.eval()

class TrackAgileModuleVer12(nn.Module):
    """
    Based on ver11
    Add av input
    """
    def __init__(self, device='cpu'):
        super(TrackAgileModuleVer12, self).__init__()
        self.device = device


        # Initialize Decision module
        self.decision_module = TrackAgileModuleVer5Dicision(input_size=9+3,device=device).to(device)

        self.predict_module = TrackTransferModuleVer0Predict(device=device).to(device)

    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load the model's state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.eval()
