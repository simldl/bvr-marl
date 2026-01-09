#!/bin/bash
# Install PyTorch with CUDA 12.8 support
# This must be run separately because PyTorch CUDA builds are not on PyPI

echo "Installing PyTorch with CUDA 12.8 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "PyTorch installation complete!"
echo "Verify with: python -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')\""
