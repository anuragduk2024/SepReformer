# SepReformer for Speech Separation [NeurIPS 2024]

This is the official implementation of "Separate and Reconstruct: Asymmetric Encoder-Decoder for Speech Separation" accepted in NeurIPS 2024. [Paper Link(Arxiv)](https://arxiv.org/abs/2406.05983)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/separate-and-reconstruct-asymmetric-encoder/speech-separation-on-wsj0-2mix)](https://paperswithcode.com/sota/speech-separation-on-wsj0-2mix?p=separate-and-reconstruct-asymmetric-encoder)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/separate-and-reconstruct-asymmetric-encoder/speech-separation-on-wham)](https://paperswithcode.com/sota/speech-separation-on-wham?p=separate-and-reconstruct-asymmetric-encoder)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/separate-and-reconstruct-asymmetric-encoder/speech-separation-on-whamr)](https://paperswithcode.com/sota/speech-separation-on-whamr?p=separate-and-reconstruct-asymmetric-encoder)

## News
- **October 2024**: Added feature for single audio inference on SepReformer-B for WSJ0-2MIX.
- **October 2024**: Uploaded pre-trained models for SepReformer-B for WSJ0-2MIX.
- **September 2024**: Paper accepted at NeurIPS 2024 🎉.

## Todo
- Release other cases for partially or fully overlapped, noisy-reverberant mixture with 16k sampling rates for practical application by the end of this year.

## Environment Preparation
To set up the environment, run the following commands:
```bash
conda create -n SepReformer python=3.10
conda activate SepReformer
pip install -r requirements.txt
```

## Pretrained Models
We offer a pretrained model for SepReformer-B. Other models will be uploaded soon. This repository uses **Git LFS (Large File Storage)** to manage pretrained model files. If Git LFS is not installed, large files may not be downloaded properly. **Please install Git LFS before cloning this repository.**

## Project Structure
- `app.py`: Main application script.
- `configs.yaml`: Configuration file for the project.
- `data/`: Directory containing datasets.
- `engine.py`: Core engine functionalities.
- `evaluate.py`: Script for evaluating the model.
- `main.py`: Entry point for running the model.
- `model/`: Contains model architecture and related scripts.
- `preprocess.py`: Script for preprocessing data.
- `train.py`: Script for training the model.
- `utils/`: Utility functions and helpers.

## Requirements
The project requires the following Python packages, which are specified in `requirements.txt`:
- `absl-py==2.1.0`
- `audioread==3.0.1`
- `certifi==2024.8.30`
- `cffi==1.17.1`
- `charset-normalizer==3.4.0`
- `contourpy==1.3.0`
- `cycler==0.12.1`
- `decorator==5.1.1`
- `filelock==3.16.1`
- `fonttools==4.54.1`
- `fsspec==2024.9.0`
- `future==1.0.0`
- `grpcio==1.67.0`
- `idna==3.10`
- `Jinja2==3.1.4`
- `joblib==1.4.2`
- `kiwisolver==1.4.7`
- `lazy_loader==0.4`
- `librosa==0.10.2.post1`
- `llvmlite==0.43.0`
- `loguru==0.7.2`
- `Markdown==3.7`
- `MarkupSafe==3.0.1`
- `matplotlib==3.9.2`
- `mir-eval==0.7`
- `mpmath==1.3.0`
- `msgpack==1.1.0`
- `networkx==3.4.1`
- `numba==0.60.0`
- `numpy==1.26.4`
- `nvidia-cublas-cu12==12.1.3.1`
- `nvidia-cuda-cupti-cu12==12.1.105`
- `nvidia-cuda-nvrtc-cu12==12.1.105`
- `nvidia-cuda-runtime-cu12==12.1.105`
- `nvidia-cudnn-cu12==8.9.2.26`
- `nvidia-cufft-cu12==11.0.2.54`
- `nvidia-curand-cu12==10.3.2.106`
- `nvidia-cusolver-cu12==11.4.5.107`
- `nvidia-cusparse-cu12==12.1.0.106`
- `nvidia-nccl-cu12==2.18.1`
- `nvidia-nvjitlink-cu12==12.6.77`
- `nvidia-nvtx-cu12==12.1.105`
- `packaging==24.1`
- `pandas==2.2.3`
- `pillow==11.0.0`
- `platformdirs==4.3.6`
- `pooch==1.8.2`
- `protobuf==5.28.2`
- `ptflops==0.7.4`
- `pycparser==2.22`
- `pyparsing==3.2.0`

## Usage
Instructions on how to use the project will be added here.

## License
This project is licensed under the terms of the license file included in the repository.

## Data Preparation

- For training or evaluation, you need dataset and scp file
    1. Prepare dataset for speech separation (eg. WSJ0-2mix)
    2. create scp file using data/create_scp/*.py

## Training

- If you want to train the network, you can simply trying by
    - set the scp file in ‘models/SepReformer_Base_WSJ0/configs.yaml’
    - run training as
        
        ```bash
        python run.py --model SepReformer_Base_WSJ0 --engine-mode train
        ```

### Inference on a single audio sample

- Simply Inference on a single audio with output wav files saved

    ```bash
    python run.py --model SepReformer_Base_WSJ0 --engine-mode infer_sample --sample-file /to/your/sample/dir/
    ```

- For example, you can directly test by using the included sample as

    ```bash
    python run.py --model SepReformer_Base_WSJ0 --engine-mode infer_sample --sample-file ./sample_wav/sample_WSJ.wav
    ```


## Test on Dataset

- Evaluating a model on dataset without saving output as audio files
    
    ```bash
    python run.py --model SepReformer_Base_WSJ0 --engine-mode test
    ```
    

- Evaluating on dataset with output wav files saved
    
    ```bash
    python run.py --model SepReformer_Base_WSJ0 --engine-mode test_wav --out_wav_dir '/your/save/directoy[optional]'
    ```
    

## Training Curve
- For SepReformer-B with WSJ-2MIX, the training and validation curve is as follows:
![Untitled](data/figure/Training_Curve.png)

<br />
<br />

![Untitled](data/figure/Result_table.png)

![Untitled](data/figure/SISNRvsMACs.png)

## Citation

If you find this repository helpful, please consider citing:
```
@inproceedings{
shin2024separate,
title={Separate and Reconstruct: Asymmetric Encoder-Decoder for Speech Separation},
author={Ui-Hyeop Shin and Sangyoun Lee and Taehan Kim and Hyung-Min Park},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=99y2EfLe3B}
}
```
