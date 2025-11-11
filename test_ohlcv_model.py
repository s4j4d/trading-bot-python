"""
Test script to verify OHLCV model architecture works correctly.
"""

import numpy as np
import tensorflow as tf
from model.q_network import build_q_network
from config.constants import WINDOW_SIZE

print("="*60)
print("TESTING OHLCV MODEL ARCHITECTURE")
print("="*60)

# Test parameters
window_size = WINDOW_SIZE  # Should be 30
num_features = 5  # OHLCV
action_size = 3  # Hold, Buy, Sell

# Calculate input shape
input_shape = (window_size * num_features,)  # (30 * 5,) = (150,)

print(f"\nConfiguration:")
print(f"  Window Size: {window_size}")
print(f"  Features: {num_features} (OHLCV)")
print(f"  Input Shape: {input_shape}")
print(f"  Actions: {action_size}")

# Build model
print(f"\nBuilding Q-Network...")
try:
    model = build_q_network(input_shape, action_size)
    print("✅ Model built successfully!")
    
    # Print model summary
    print(f"\nModel Summary:")
    model.summary()
    
    # Test with dummy data
    print(f"\nTesting with dummy data...")
    dummy_input = np.random.randn(1, window_size * num_features).astype(np.float32)
    print(f"  Input shape: {dummy_input.shape}")
    
    # Make prediction
    q_values = model.predict(dummy_input, verbose=0)
    print(f"  Output shape: {q_values.shape}")
    print(f"  Q-values: {q_values[0]}")
    print(f"  Best action: {np.argmax(q_values[0])} ({'Hold' if np.argmax(q_values[0])==0 else 'Buy' if np.argmax(q_values[0])==1 else 'Sell'})")
    
    print(f"\n✅ All tests passed!")
    print(f"\nThe model is ready to use OHLCV features!")
    print(f"Next step: Retrain the model with:")
    print(f"  python training/trainer.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("="*60)
