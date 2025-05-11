from preprocess import get_random_wav_batch, load_wavs, wavs_to_specs, sample_data_batch, sperate_magnitude_phase
from model import SVSRNN
import os
import librosa
import numpy as np
import tensorflow as tf
import glob

def train(random_seed = 0):
    np.random.seed(random_seed)

    # MIR-1K dataset directory
    mir1k_dir = 'MIR-1K/Wavfile'
    
    # Get all wav files from MIR-1K dataset
    wav_files = glob.glob(os.path.join(mir1k_dir, '*.wav'))
    np.random.shuffle(wav_files)
    
    # Split into training and validation sets (90-10 split)
    split_idx = int(len(wav_files) * 0.9)
    wav_filenames_train = wav_files[:split_idx]
    wav_filenames_valid = wav_files[split_idx:]
    
    # Save train/valid splits for reproducibility
    os.makedirs('MIR-1K', exist_ok=True)
    with open('MIR-1K/train.txt', 'w') as f:
        f.write('\n'.join(wav_filenames_train))
    with open('MIR-1K/valid.txt', 'w') as f:
        f.write('\n'.join(wav_filenames_valid))

    # Preprocess parameters
    mir1k_sr = 16000  # MIR-1K sampling rate
    n_fft = 1024
    hop_length = n_fft // 4

    # Model parameters
    learning_rate = 0.0001
    num_rnn_layer = 4
    num_hidden_units = [256, 256, 256, 256]
    batch_size = 32
    sample_frames = 128  # Increased for better context
    iterations = 5000
    tensorboard_directory = './graphs/svsrnn'
    log_directory = './log'
    train_log_filename = 'train_log.csv'
    clear_tensorboard = False
    model_directory = './model'
    model_filename = 'svsrnn.weights.h5'

    # Create necessary directories
    os.makedirs(log_directory, exist_ok=True)
    os.makedirs(model_directory, exist_ok=True)
    os.makedirs(tensorboard_directory, exist_ok=True)
    
    # Initialize log file
    with open(os.path.join(log_directory, train_log_filename), 'w') as f:
        f.write('iteration,train_loss,validation_loss\n')

    print(f"Loading {len(wav_filenames_train)} training files and {len(wav_filenames_valid)} validation files...")

    # Load train wavs
    wavs_mono_train, wavs_src1_train, wavs_src2_train = load_wavs(filenames=wav_filenames_train, sr=mir1k_sr)
    print("Converting training files to spectrograms...")
    stfts_mono_train, stfts_src1_train, stfts_src2_train = wavs_to_specs(
        wavs_mono=wavs_mono_train, wavs_src1=wavs_src1_train, wavs_src2=wavs_src2_train, 
        n_fft=n_fft, hop_length=hop_length)

    # Load validation wavs
    print("Loading validation files...")
    wavs_mono_valid, wavs_src1_valid, wavs_src2_valid = load_wavs(filenames=wav_filenames_valid, sr=mir1k_sr)
    print("Converting validation files to spectrograms...")
    stfts_mono_valid, stfts_src1_valid, stfts_src2_valid = wavs_to_specs(
        wavs_mono=wavs_mono_valid, wavs_src1=wavs_src1_valid, wavs_src2=wavs_src2_valid,
        n_fft=n_fft, hop_length=hop_length)

    print("Initializing model...")
    model = SVSRNN(
        num_features=n_fft // 2 + 1,
        num_rnn_layer=num_rnn_layer,
        num_hidden_units=num_hidden_units,
        tensorboard_directory=tensorboard_directory,
        clear_tensorboard=clear_tensorboard
    )

    print("Starting training...")
    best_valid_loss = float('inf')
    patience = 10  # Early stopping patience
    patience_counter = 0

    # Start training
    for i in range(iterations):
        # Training step
        data_mono_batch, data_src1_batch, data_src2_batch = sample_data_batch(
            stfts_mono=stfts_mono_train, stfts_src1=stfts_src1_train, stfts_src2=stfts_src2_train,
            batch_size=batch_size, sample_frames=sample_frames)
        
        x_mixed, _ = sperate_magnitude_phase(data=data_mono_batch)
        y1, _ = sperate_magnitude_phase(data=data_src1_batch)
        y2, _ = sperate_magnitude_phase(data=data_src2_batch)

        # Ensure inputs are properly shaped [batch_size, time_steps, features]
        x_mixed = np.expand_dims(x_mixed, axis=-1) if x_mixed.ndim == 2 else x_mixed
        y1 = np.expand_dims(y1, axis=-1) if y1.ndim == 2 else y1
        y2 = np.expand_dims(y2, axis=-1) if y2.ndim == 2 else y2

        train_loss = model.train(x=x_mixed, y1=y1, y2=y2, learning_rate=learning_rate)

        if i % 10 == 0:
            print(f'Step: {i} Train Loss: {train_loss:.6f}')

        # Validation step
        if i % 200 == 0:
            print('=' * 50)
            data_mono_batch, data_src1_batch, data_src2_batch = sample_data_batch(
                stfts_mono=stfts_mono_valid, stfts_src1=stfts_src1_valid, stfts_src2=stfts_src2_valid,
                batch_size=batch_size, sample_frames=sample_frames)
            
            x_mixed, _ = sperate_magnitude_phase(data=data_mono_batch)
            y1, _ = sperate_magnitude_phase(data=data_src1_batch)
            y2, _ = sperate_magnitude_phase(data=data_src2_batch)

            # Ensure validation inputs are properly shaped
            x_mixed = np.expand_dims(x_mixed, axis=-1) if x_mixed.ndim == 2 else x_mixed
            y1 = np.expand_dims(y1, axis=-1) if y1.ndim == 2 else y1
            y2 = np.expand_dims(y2, axis=-1) if y2.ndim == 2 else y2

            y1_pred, y2_pred, validation_loss = model.validate(x=x_mixed, y1=y1, y2=y2)
            print(f'Step: {i} Validation Loss: {validation_loss:.6f}')
            print('=' * 50)

            # Save training log
            with open(os.path.join(log_directory, train_log_filename), 'a') as log_file:
                log_file.write(f'{i},{train_loss},{validation_loss}\n')

            # Early stopping check
            if validation_loss < best_valid_loss:
                best_valid_loss = validation_loss
                patience_counter = 0
                # Save best model
                model.save(directory=model_directory, filename=model_filename)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {i} iterations")
                    break

        # Regular model checkpoint
        if i % 1000 == 0:
            model.save(directory=model_directory, filename=f'svsrnn_checkpoint_{i}.weights.h5')

    print("Training completed!")

if __name__ == '__main__':
    train()
