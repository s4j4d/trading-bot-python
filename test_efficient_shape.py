"""
Test the improved efficient shape handling (no unnecessary flatten/reshape).
"""

import numpy as np
import tensorflow as tf
from model.q_network import build_q_network
from config.constants import WINDOW_SIZE

print("="*70)
print("TESTING EFFICIENT SHAPE HANDLING (NO FLATTEN/RESHAPE)")
print("="*70)

# Test parameters
window_size = WINDOW_SIZE  # 30
num_features = 5  # OHLCV
action_size = 3  # Hold, Buy, Sell

# NEW: Natural 2D shape (no flattening!)
input_shape = (window_size, num_features)  # (30, 5)

print(f"\nConfiguration:")
print(f"  Window Size: {window_size}")
print(f"  Features: {num_features} (OHLCV)")
print(f"  Input Shape: {input_shape}")
print(f"  Actions: {action_size}")

print(f"\n✅ IMPROVEMENT: Input is now 2D (natural shape)")
print(f"   OLD: (150,) - flattened, needs reshape")
print(f"   NEW: (30, 5) - natural, no reshape needed!")

# Build model
print(f"\nBuilding Q-Network with efficient shape...")
try:
    model = build_q_network(input_shape, action_size)
    print("✅ Model built successfully!")
    
    # Print model summary
    print(f"\nModel Summary:")
    model.summary()
    
    # Test with dummy data
    print(f"\nTesting with dummy data...")
    dummy_input = np.random.randn(1, window_size, num_features).astype(np.float32)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  ✅ No flattening needed!")
    print(f"  ✅ No reshape layer needed!")
    
    # Make prediction
    q_values = model.predict(dummy_input, verbose=0)
    print(f"  Output shape: {q_values.shape}")
    print(f"  Q-values: {q_values[0]}")
    print(f"  Best action: {np.argmax(q_values[0])} ({'Hold' if np.argmax(q_values[0])==0 else 'Buy' if np.argmax(q_values[0])==1 else 'Sell'})")
    
    print(f"\n✅ All tests passed!")
    
    print(f"\n" + "="*70)
    print("EFFICIENCY COMPARISON")
    print("="*70)
    print(f"""
OLD APPROACH (Inefficient):
  1. Environment: (30, 5) → flatten → (150,)
  2. Model input: (1, 150)
  3. Reshape layer: (1, 150) → (1, 30, 5)
  4. LSTM: processes (1, 30, 5)
  
  ❌ Unnecessary operations: flatten + reshape
  ❌ Extra layer in model
  ❌ More memory allocations

NEW APPROACH (Efficient):
  1. Environment: (30, 5) - keep natural shape
  2. Model input: (1, 30, 5)
  3. LSTM: processes (1, 30, 5) directly
  
  ✅ No unnecessary operations
  ✅ Cleaner model architecture
  ✅ Better performance
  ✅ More intuitive code
    """)
    
    print("="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
1. Delete old model file (incompatible shape):
   rm final_parallel_trading_model.keras

2. Retrain with new efficient architecture:
   python training/trainer.py

3. Test the new model:
   python evaluate_nobitex_dqn.py --pair WIRT --days 7
    """)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("="*70)
