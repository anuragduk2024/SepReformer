import os
import torch
import numpy as np
import yaml
from datetime import datetime
from model import SVSRNN
from preprocess import load_wavs, wavs_to_specs, sample_data_batch, sperate_magnitude_phase
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Engine:
    def __init__(self, config_path="configs.yaml", device="cuda" if torch.cuda.is_available() else "cpu"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = device
        # Initialize model
        self.model = SVSRNN(config_path).to(device)
        
        # Setup directories
        self.setup_directories()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Initialize tensorboard
        self.writer = SummaryWriter(os.path.join('logs', 'svsrnn'))
        
    def setup_directories(self):
        """Create necessary directories for logging and checkpoints"""
        dirs = ['logs', 'checkpoints', 'results']
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)
            
    def train(self, train_data, valid_data):
        """Main training loop"""
        config = self.config['training']
        
        for epoch in range(config['num_epochs']):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
            
            # Training
            train_loss = self.train_epoch(train_data)
            print(f"Training Loss: {train_loss:.6f}")
            
            # Log training loss
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            
            # Validation
            if (epoch + 1) % config['validation_interval'] == 0:
                valid_loss = self.validate(valid_data)
                print(f"Validation Loss: {valid_loss:.6f}")
                
                # Log validation loss
                self.writer.add_scalar('Loss/valid', valid_loss, epoch)
                
                # Early stopping check
                if valid_loss < self.best_loss - config['early_stopping']['min_delta']:
                    self.best_loss = valid_loss
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= config['early_stopping']['patience']:
                    print("Early stopping triggered!")
                    break
            
            # Regular checkpoint saving
            if (epoch + 1) % config['checkpoint']['save_interval'] == 0:
                self.save_checkpoint()
                
    def train_epoch(self, data):
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(data, desc='Training')
        for batch in pbar:
            # Move data to device
            x_mixed = torch.from_numpy(batch['mixed']).to(self.device)
            y1 = torch.from_numpy(batch['source1']).to(self.device)
            y2 = torch.from_numpy(batch['source2']).to(self.device)
            
            loss = self.model.train_step(x_mixed, [y1, y2])
            total_loss += loss
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss:.6f}'})
            
        return total_loss / num_batches
    
    def validate(self, data):
        """Validation step"""
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(data, desc='Validation')
        for batch in pbar:
            # Move data to device
            x_mixed = torch.from_numpy(batch['mixed']).to(self.device)
            y1 = torch.from_numpy(batch['source1']).to(self.device)
            y2 = torch.from_numpy(batch['source2']).to(self.device)
            
            loss = self.model.validate(x_mixed, [y1, y2])
            total_loss += loss
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss:.6f}'})
            
        return total_loss / num_batches
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = 'checkpoints'
        if is_best:
            path = os.path.join(checkpoint_dir, 'best_model')
        else:
            path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{self.current_epoch}')
            
        self.model.save_checkpoint(path, self.current_epoch, is_best)
        
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        self.current_epoch = self.model.load_checkpoint(path)
        
    def prepare_data(self, wav_files, batch_size=None):
        """Prepare data for training/validation"""
        if batch_size is None:
            batch_size = self.config['training']['batch_size']
            
        # Load and preprocess audio files
        wavs_mono, wavs_src1, wavs_src2 = load_wavs(
            filenames=wav_files,
            sr=self.config['data']['sample_rate']
        )
        
        # Convert to spectrograms
        specs_mono, specs_src1, specs_src2 = wavs_to_specs(
            wavs_mono=wavs_mono,
            wavs_src1=wavs_src1,
            wavs_src2=wavs_src2,
            n_fft=self.config['data']['n_fft'],
            hop_length=self.config['data']['hop_length']
        )
        
        def data_generator():
            while True:
                data_mono_batch, data_src1_batch, data_src2_batch = sample_data_batch(
                    stfts_mono=specs_mono,
                    stfts_src1=specs_src1,
                    stfts_src2=specs_src2,
                    batch_size=batch_size,
                    sample_frames=128  # Using larger context window
                )
                
                yield {
                    'mixed': data_mono_batch,
                    'source1': data_src1_batch,
                    'source2': data_src2_batch
                }
                
        return data_generator()
