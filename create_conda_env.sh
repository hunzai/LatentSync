#!/bin/bash

echo "Setting up LatentSync environment for RTX 5090..."

# Create a new conda environment
conda create -y -n latentsync python=3.10.13

# Activate the environment
source activate latentsync

# Install ffmpeg
conda install -y -c conda-forge ffmpeg

# Install PyTorch 2.7.1 with CUDA 12.8 support for RTX 5090
echo "Installing PyTorch 2.7.1 with CUDA 12.8 support..."
pip install --root-user-action=ignore torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Python dependencies
echo "Installing Python packages..."
pip install --root-user-action=ignore -r requirements.txt

# OpenCV dependencies
sudo apt -y install libgl1

# Download the checkpoints required for inference from HuggingFace
echo "Downloading model checkpoints..."
mkdir -p checkpoints

# Download Whisper model
echo "Downloading Whisper model..."
huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints

# Download LatentSync U-Net model
echo "Downloading LatentSync U-Net model..."
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints

# Test PyTorch installation
echo "Testing PyTorch installation..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA version:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); x = torch.randn(100, 100).cuda(); print('GPU test: SUCCESS')"

echo "LatentSync environment setup completed successfully!"
echo "To activate: conda activate latentsync"
echo "To run: python gradio_app.py"