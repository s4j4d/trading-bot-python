"""
Detailed explanation of input shapes through the model.
"""

import numpy as np
from config.constants import WINDOW_SIZE

print("="*70)
print("INPUT SHAPE FLOW THROUGH THE MODEL")
print("="*70)

# Configuration
window_size = WINDOW_SIZE  # 30
num_features = 5  # OHLCV (Open, High, Low, Close, Volume)
batch_size = 1  # For single prediction

print(f"\nConfiguration:")
print(f"  WINDOW_SIZE: {window_size}")
print(f"  NUM_FEATURES: {num_features} (OHLCV)")
print(f"  BATCH_SIZE: {batch_size}")

print("\n" + "="*70)
print("STEP-BY-STEP DATA FLOW")
print("="*70)

# Step 1: Raw OHLCV data from environment
print("\n1. RAW DATA FROM ENVIRONMENT")
print("-" * 70)
raw_ohlcv = np.random.randn(window_size, num_features)
print(f"   Shape: {raw_ohlcv.shape}")
print(f"   Meaning: {window_size} timesteps × {num_features} features")
print(f"   Structure:")
print(f"     Row 0: [open_0, high_0, low_0, close_0, volume_0]")
print(f"     Row 1: [open_1, high_1, low_1, close_1, volume_1]")
print(f"     ...")
print(f"     Row {window_size-1}: [open_{window_size-1}, high_{window_size-1}, low_{window_size-1}, close_{window_size-1}, volume_{window_size-1}]")
print(f"\n   Example (first 3 rows):")
print(raw_ohlcv[:3])

# Step 2: Flatten for model input
print("\n2. FLATTEN FOR MODEL INPUT")
print("-" * 70)
flattened = raw_ohlcv.flatten()
print(f"   Shape: {flattened.shape}")
print(f"   Meaning: All {window_size * num_features} values in a single row")
print(f"   Structure: [o0,h0,l0,c0,v0, o1,h1,l1,c1,v1, ..., o{window_size-1},h{window_size-1},l{window_size-1},c{window_size-1},v{window_size-1}]")
print(f"\n   Example (first 15 values):")
print(flattened[:15])

# Step 3: Add batch dimension
print("\n3. ADD BATCH DIMENSION")
print("-" * 70)
batched = np.expand_dims(flattened, axis=0)
print(f"   Shape: {batched.shape}")
print(f"   Meaning: {batch_size} sample(s) × {window_size * num_features} features")
print(f"   This is the INPUT to the model!")
print(f"\n   Example (first 15 values of batch):")
print(batched[0, :15])

# Step 4: Input Reshape Layer
print("\n4. INPUT_RESHAPE LAYER (inside model)")
print("-" * 70)
print(f"   Input shape:  {batched.shape}")
print(f"   Reshape to:   ({batch_size}, {window_size}, {num_features})")
reshaped = batched.reshape(batch_size, window_size, num_features)
print(f"   Output shape: {reshaped.shape}")
print(f"   Meaning: {batch_size} sample(s) × {window_size} timesteps × {num_features} features")
print(f"\n   Structure after reshape:")
print(f"     Timestep 0: [open_0, high_0, low_0, close_0, volume_0]")
print(f"     Timestep 1: [open_1, high_1, low_1, close_1, volume_1]")
print(f"     ...")
print(f"     Timestep {window_size-1}: [open_{window_size-1}, high_{window_size-1}, low_{window_size-1}, close_{window_size-1}, volume_{window_size-1}]")
print(f"\n   Example (first 3 timesteps):")
print(reshaped[0, :3, :])

# Step 5: LSTM Layer 1
print("\n5. LSTM LAYER 1")
print("-" * 70)
print(f"   Input shape:  {reshaped.shape}")
print(f"   LSTM units:   64")
print(f"   return_sequences: True")
print(f"   Output shape: ({batch_size}, {window_size}, 64)")
print(f"   Meaning: {batch_size} sample(s) × {window_size} timesteps × 64 hidden features")

# Step 6: LSTM Layer 2
print("\n6. LSTM LAYER 2")
print("-" * 70)
print(f"   Input shape:  ({batch_size}, {window_size}, 64)")
print(f"   LSTM units:   32")
print(f"   return_sequences: False")
print(f"   Output shape: ({batch_size}, 32)")
print(f"   Meaning: {batch_size} sample(s) × 32 hidden features (final state only)")

# Step 7: Dense Layer
print("\n7. DENSE LAYER")
print("-" * 70)
print(f"   Input shape:  ({batch_size}, 32)")
print(f"   Dense units:  64")
print(f"   Output shape: ({batch_size}, 64)")

# Step 8: Output Layer
print("\n8. OUTPUT LAYER (Q-VALUES)")
print("-" * 70)
print(f"   Input shape:  ({batch_size}, 64)")
print(f"   Dense units:  3 (Hold, Buy, Sell)")
print(f"   Output shape: ({batch_size}, 3)")
print(f"   Meaning: {batch_size} sample(s) × 3 Q-values")
print(f"\n   Example output: [Q_hold, Q_buy, Q_sell]")
print(f"                   [0.123,   0.456,  -0.789]")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
The input_reshape layer receives:
  Input:  (batch_size, {window_size * num_features})  = ({batch_size}, {window_size * num_features})
  Output: (batch_size, {window_size}, {num_features})  = ({batch_size}, {window_size}, {num_features})

This transforms the flattened OHLCV data back into a 3D tensor
that LSTM can process as a sequence of {window_size} timesteps,
each with {num_features} features (OHLCV).
""")

print("\n" + "="*70)
print("PRACTICAL EXAMPLE")
print("="*70)
print(f"""
When you call model.predict():

1. You provide: shape ({batch_size}, {window_size * num_features})
   Example: np.array([[o0,h0,l0,c0,v0, o1,h1,l1,c1,v1, ..., o29,h29,l29,c29,v29]])

2. input_reshape converts to: shape ({batch_size}, {window_size}, {num_features})
   Example: np.array([
       [[o0,h0,l0,c0,v0],
        [o1,h1,l1,c1,v1],
        ...
        [o29,h29,l29,c29,v29]]
   ])

3. LSTM processes this as {window_size} sequential timesteps
   Each timestep has {num_features} features

4. Final output: shape ({batch_size}, 3)
   Example: np.array([[0.123, 0.456, -0.789]])  # Q-values for Hold, Buy, Sell
""")

print("="*70)
