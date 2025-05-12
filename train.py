import torch
import os
import glob
import numpy as np
from tqdm import tqdm
import librosa
from model import SVSRNN
from preprocess import load_wavs, wavs_to_specs, sperate_magnitude_phase
import json
from datetime import datetime

def segment_spectrograms(specs, segment_size=128):
    """Segment spectrograms into fixed-length chunks"""
    segmented = []
    for spec in specs:
        # Get number of complete segments
        num_segments = spec.shape[1] // segment_size
        if num_segments == 0:
            # Pad if shorter than segment_size
            padded = np.pad(spec, ((0, 0), (0, segment_size - spec.shape[1])), mode='constant')
            segmented.append(padded)
        else:
            # Split into segments
            for i in range(num_segments):
                start = i * segment_size
                end = start + segment_size
                segmented.append(spec[:, start:end])
            # Handle remaining frames if needed
            if spec.shape[1] % segment_size != 0:
                last_segment = spec[:, -segment_size:]
                segmented.append(last_segment)
    return segmented

def train(random_seed=0):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset parameters
    mir1k_dir = 'MIR-1K/Wavfile'
    mir1k_sr = 16000
    n_fft = 1024
    hop_length = n_fft // 4
    segment_size = 128  # Number of time frames per segment
    
    # Model parameters
    num_features = n_fft // 2 + 1
    num_rnn_layer = 4
    num_hidden_units = [256, 256, 256, 256]
    batch_size = 32
    num_epochs = 100
    
    # Create necessary directories
    os.makedirs('model', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Initialize logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join('logs', f'training_log_{timestamp}.json')
    training_history = []
    
    # Get all wav files from MIR-1K dataset
    wav_files = glob.glob(os.path.join(mir1k_dir, '*.wav'))
    if not wav_files:
        print(f"No wav files found in {mir1k_dir}")
        return
        
    print(f"Found {len(wav_files)} wav files")
    np.random.shuffle(wav_files)
    
    # Split into training and validation sets (90-10 split)
    split_idx = int(len(wav_files) * 0.9)
    train_files = wav_files[:split_idx]
    valid_files = wav_files[split_idx:]
    
    print("Loading and preprocessing training data...")
    # Load and preprocess training data
    wavs_mono_train, wavs_src1_train, wavs_src2_train = [], [], []
    for wav_file in tqdm(train_files):
        # Load stereo audio (vocals in left channel, accompaniment in right)
        wav, _ = librosa.load(wav_file, sr=mir1k_sr, mono=False)
        if wav.shape[0] != 2:  # Skip if not stereo
            continue
        # Split channels
        wav_src1 = wav[0]  # vocals (left)
        wav_src2 = wav[1]  # accompaniment (right)
        wav_mono = wav_src1 + wav_src2  # mixed
        
        wavs_mono_train.append(wav_mono)
        wavs_src1_train.append(wav_src1)
        wavs_src2_train.append(wav_src2)
    
    print("Converting to spectrograms...")
    # Convert to spectrograms
    specs_mono_train = []
    specs_src1_train = []
    specs_src2_train = []
    
    for mono, src1, src2 in tqdm(zip(wavs_mono_train, wavs_src1_train, wavs_src2_train), desc="Computing STFTs"):
        # Compute STFTs
        spec_mono = librosa.stft(mono, n_fft=n_fft, hop_length=hop_length)
        spec_src1 = librosa.stft(src1, n_fft=n_fft, hop_length=hop_length)
        spec_src2 = librosa.stft(src2, n_fft=n_fft, hop_length=hop_length)
        
        specs_mono_train.append(spec_mono)
        specs_src1_train.append(spec_src1)
        specs_src2_train.append(spec_src2)
    
    print("Segmenting spectrograms...")
    # Segment spectrograms into fixed-length chunks
    specs_mono_segments = segment_spectrograms(specs_mono_train, segment_size)
    specs_src1_segments = segment_spectrograms(specs_src1_train, segment_size)
    specs_src2_segments = segment_spectrograms(specs_src2_train, segment_size)
    
    print(f"Total segments: {len(specs_mono_segments)}")
    
    # Initialize model
    model = SVSRNN(
        num_features=num_features,
        num_rnn_layer=num_rnn_layer,
        num_hidden_units=num_hidden_units
    ).to(device)
    
    # Training loop
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle segments for each epoch
        indices = np.random.permutation(len(specs_mono_segments))
        
        # Training
        progress_bar = tqdm(range(0, len(specs_mono_segments), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i in progress_bar:
            batch_end = min(i + batch_size, len(specs_mono_segments))
            batch_indices = indices[i:batch_end]
            
            # Get magnitude spectrograms
            x_mag = np.stack([np.abs(specs_mono_segments[j]) for j in batch_indices])
            y1_mag = np.stack([np.abs(specs_src1_segments[j]) for j in batch_indices])
            y2_mag = np.stack([np.abs(specs_src2_segments[j]) for j in batch_indices])
            
            # Convert to tensors
            x = torch.from_numpy(x_mag).float().to(device)
            y1 = torch.from_numpy(y1_mag).float().to(device)
            y2 = torch.from_numpy(y2_mag).float().to(device)
            
            # Forward pass
            loss = model.train_step(x, [y1, y2])
            total_loss += loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss:.6f}'})
            
        avg_loss = total_loss / num_batches
        
        # Log training progress
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save training log
        with open(log_file, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            print("Saving best model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'loss': best_loss,
            }, 'model/svsrnn_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
    
    print("Training completed!")
    print(f"Training history saved to {log_file}")
    print(f"Best model saved to model/svsrnn_model.pt")

if __name__ == '__main__':
    train()
