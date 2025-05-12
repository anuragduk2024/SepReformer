import torch
import torch.nn as nn
import os
import yaml
from datetime import datetime
import numpy as np

class SVSRNN(nn.Module):
    def __init__(self, num_features, num_rnn_layer, num_hidden_units, config_path=None):
        super(SVSRNN, self).__init__()
        
        self.num_features = num_features
        self.num_rnn_layer = num_rnn_layer
        self.num_spks = 2  # Fixed for vocals and background
        
        # Calculate padding for Conv1d to achieve 'same' padding effect
        encoder_padding = (16 - 1) // 2  # For kernel_size=16
        
        # Audio Encoder
        self.audio_encoder = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=16,
            stride=8,
            padding=encoder_padding,
            bias=True
        )
        
        # Feature Projector
        self.feature_projector = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_hidden_units[0],
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        # RNN Separator
        self.rnn_layers = nn.ModuleList([
            nn.GRU(
                input_size=num_hidden_units[i-1] if i > 0 else num_hidden_units[0],
                hidden_size=num_hidden_units[i],
                batch_first=True
            )
            for i in range(num_rnn_layer)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(num_hidden_units[i]) for i in range(num_rnn_layer)])
        
        # Output Layers
        self.output_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=num_hidden_units[-1],
                out_channels=num_features,
                kernel_size=1,
                padding=0,
            )
            for _ in range(self.num_spks)
        ])
        
        # Audio Decoder
        self.audio_decoder = nn.ConvTranspose1d(
            in_channels=num_features,
            out_channels=1,
            kernel_size=16,
            stride=8,
            padding=4,
            bias=True
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=0.0001,
            weight_decay=0.01
        )
        
    def forward(self, x):
        """Forward pass through the network"""
        # Convert input to torch tensor if it's not already
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Ensure input is float
        x = x.float()
        
        # Input shape should be [batch, time, features]
        # Convert to [batch, features, time] for Conv1d
        if x.dim() == 3 and x.size(2) == self.num_features:
            x = x.transpose(1, 2)
        
        # Move to the same device as the model
        x = x.to(next(self.parameters()).device)
        
        # Audio Encoding
        encoded = self.audio_encoder(x)  # [batch, features, time]
        
        # Feature Projection
        features = self.feature_projector(encoded)  # [batch, hidden, time]
        
        # RNN Processing
        x = features.transpose(1, 2)  # [batch, time, hidden]
        for rnn_layer, norm_layer in zip(self.rnn_layers, self.layer_norms):
            x, _ = rnn_layer(x)
            x = norm_layer(x)
        x = x.transpose(1, 2)  # [batch, hidden, time]
        
        # Mask Estimation
        masks = []
        for output_layer in self.output_layers:
            mask = torch.relu(output_layer(x))  # [batch, features, time]
            masks.append(mask)
        
        # Stack masks along new dimension
        mask_stack = torch.stack(masks, dim=1)  # [batch, num_spks, features, time]
        
        # Softmax across speakers (dim=1)
        mask_stack = torch.softmax(mask_stack, dim=1)
        
        # Apply masks and decode
        separated = []
        for i in range(self.num_spks):
            # Get mask for current speaker
            mask = mask_stack[:, i]  # [batch, features, time]
            
            # Apply mask to encoded features
            masked = mask * encoded  # [batch, features, time]
            
            # Decode to waveform
            decoded = self.audio_decoder(masked)  # [batch, 1, time]
            separated.append(decoded)
        
        return separated
    
    def sisnr(self, estimate, target, eps=1e-8):
        """Scale-invariant SNR loss"""
        # Normalize
        target = target - torch.mean(target, dim=-1, keepdim=True)
        estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
        
        # Compute SNR
        target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
        dot = torch.sum(target * estimate, dim=-1, keepdim=True)
        proj = dot * target / target_energy
        noise = estimate - proj
        
        ratio = (torch.sum(proj ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps)
        snr = 10 * torch.log10(ratio)
        
        return torch.mean(snr)
    
    def loss_fn(self, estimates, targets):
        """Calculate loss"""
        losses = []
        for est, target in zip(estimates, targets):
            losses.append(-self.sisnr(est, target))
        return torch.mean(torch.stack(losses))
    
    def train_step(self, x, y):
        """Single training step"""
        self.train()
        self.optimizer.zero_grad()
        
        estimates = self(x)
        loss = self.loss_fn(estimates, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)  # Fixed clip norm
        
        self.optimizer.step()
        return loss.item()
    
    def validate(self, x, y):
        """Validation step"""
        self.eval()
        with torch.no_grad():
            estimates = self(x)
            loss = self.loss_fn(estimates, y)
        return loss.item()
    
    def save_checkpoint(self, path, epoch, is_best=False):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{path}.pt")
        if is_best:
            best_path = os.path.join(os.path.dirname(path), 'best_model.pt')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    def load(self, filepath):
        """Load model weights from a file"""
        if filepath.endswith('.h5'):  # TensorFlow weights
            print("Warning: TensorFlow weights detected. Please convert weights to PyTorch format first.")
            return
            
        if not os.path.exists(filepath):
            print(f"Warning: Model file {filepath} not found.")
            return
            
        try:
            checkpoint = torch.load(filepath, map_location=next(self.parameters()).device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Loading from our own checkpoint format
                self.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                # Direct state dict
                self.load_state_dict(checkpoint)
            print(f"Successfully loaded model from {filepath}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            
    def test(self, x):
        """Run inference on input data"""
        self.eval()
        with torch.no_grad():
            separated = self(x)
            # Convert to numpy arrays
            return [s.cpu().numpy() for s in separated]