#!/usr/bin/env python3
"""
Quick script to check GPU availability and TensorFlow GPU setup.
"""

import tensorflow as tf
import sys

print("TensorFlow version:", tf.__version__)
print("Python version:", sys.version)
print()

# Check if GPU is available
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())

# List all physical devices
physical_devices = tf.config.list_physical_devices()
print("\nAll physical devices:")
for device in physical_devices:
    print(f"  {device}")

# Check GPU memory growth setting
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\nFound {len(gpus)} GPU(s)")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
        
    # Test GPU computation
    print("\nTesting GPU computation...")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("GPU computation successful!")
            print("Result shape:", c.shape)
    except Exception as e:
        print(f"GPU computation failed: {e}")
else:
    print("\nNo GPUs found. Training will use CPU.")
    print("To enable GPU:")
    print("1. Install CUDA and cuDNN")
    print("2. Install tensorflow-gpu or tensorflow with GPU support")