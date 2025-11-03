#!/bin/bash
"""
TensorFlow GPU Installation Script for RTX 5080

This script installs the correct TensorFlow version with CUDA support
for RTX 5080 (compute capability 12.0).
"""

echo "Installing TensorFlow with GPU support for RTX 5080..."
echo "=================================================="

# Remove existing TensorFlow installations
echo "Removing existing TensorFlow installations..."
pip uninstall -y tensorflow tensorflow-gpu tf-nightly

# Install latest TensorFlow with CUDA support
echo "Installing latest TensorFlow with CUDA support..."
pip install tensorflow[and-cuda]

# If that fails, try the nightly build
if [ $? -ne 0 ]; then
    echo "Trying TensorFlow nightly build..."
    pip install tf-nightly[and-cuda]
fi

# Verify installation
echo "Verifying installation..."
python3 -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'CUDA support: {tf.test.is_built_with_cuda()}')
print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')
"

echo "Installation complete!"
echo "Now try running: python3 main_gpu_fixed.py"