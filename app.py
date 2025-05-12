import streamlit as st
import os
import tempfile
import librosa
import numpy as np
import torch
import soundfile as sf
from preprocess import sperate_magnitude_phase, combine_magnitdue_phase
from model import SVSRNN

# Assuming you have the necessary imports and functions from your original code

def separate_sources(audio_file):
    # Preprocess parameters
    mir1k_sr = 16000
    n_fft = 1024
    hop_length = n_fft // 4
    num_rnn_layer = 4
    num_hidden_units = [256, 256, 256, 256]
    model_directory = 'model'
    model_filename = 'svsrnn_model.pt'  # Changed to .pt extension for PyTorch
    model_filepath = os.path.join(model_directory, model_filename)

    # Load and preprocess audio
    wav_mono, _ = librosa.load(audio_file, sr=mir1k_sr, mono=True)
    stft_mono = librosa.stft(wav_mono, n_fft=n_fft, hop_length=hop_length)
    stft_mono = stft_mono.transpose()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SVSRNN(num_features=n_fft // 2 + 1, num_rnn_layer=num_rnn_layer, num_hidden_units=num_hidden_units).to(device)
    
    # Build model with a sample input before loading weights
    sample_input = np.zeros((1, stft_mono.shape[0], n_fft // 2 + 1))
    _ = model(sample_input)  # This builds the model
    
    if os.path.exists(model_filepath):
        print(f"Loading model weights from {model_filepath}")
        checkpoint = torch.load(model_filepath, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: No model weights found at {model_filepath}")
        return None, None

    # Process audio
    stft_mono_magnitude, stft_mono_phase = sperate_magnitude_phase(data=stft_mono)
    stft_mono_magnitude = np.array([stft_mono_magnitude])

    # Get predictions
    model.eval()
    with torch.no_grad():
        y1_pred, y2_pred = model.test(stft_mono_magnitude)

    # ISTFT with the phase from mono
    y1_stft_hat = combine_magnitdue_phase(magnitudes=y1_pred[0], phases=stft_mono_phase)
    y2_stft_hat = combine_magnitdue_phase(magnitudes=y2_pred[0], phases=stft_mono_phase)

    # Transpose if needed
    if y1_stft_hat.ndim == 2:
        y1_stft_hat = y1_stft_hat.T
        y2_stft_hat = y2_stft_hat.T

    # Convert to waveform
    y1_hat = librosa.istft(y1_stft_hat, hop_length=hop_length)
    y2_hat = librosa.istft(y2_stft_hat, hop_length=hop_length)

    return y1_hat, y2_hat

def main():
    st.title("Audio Source Separation App")
    st.write("Upload an MP3 file to separate the vocals from the background music.")

    uploaded_file = st.file_uploader("Choose an MP3 file", type="mp3")

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/mp3')

        if st.button("Separate Audio"):
            with st.spinner("Processing audio..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_filepath = tmp_file.name

                # Process the audio
                vocals, background = separate_sources(tmp_filepath)
                
                if vocals is None or background is None:
                    st.error("Error: Could not process audio. Model weights not found.")
                else:
                    # Save processed audio files
                    vocal_path = "vocals.wav"
                    background_path = "background.wav"
                    sf.write(vocal_path, vocals, 16000)
                    sf.write(background_path, background, 16000)

                    # Clean up temporary file
                    os.unlink(tmp_filepath)

                    st.success("Audio separated successfully!")

                    # Display and allow download of separated audio
                    st.subheader("Separated Vocals")
                    st.audio(vocal_path, format='audio/wav')
                    with open(vocal_path, "rb") as file:
                        st.download_button(
                            label="Download Vocals",
                            data=file,
                            file_name="vocals.wav",
                            mime="audio/wav"
                        )

                    st.subheader("Separated Background Music")
                    st.audio(background_path, format='audio/wav')
                    with open(background_path, "rb") as file:
                        st.download_button(
                            label="Download Background Music",
                            data=file,
                            file_name="background.wav",
                            mime="audio/wav"
                        )

if __name__ == "__main__":
    main()