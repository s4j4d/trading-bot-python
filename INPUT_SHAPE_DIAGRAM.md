# Input Shape Flow - Visual Diagram

## Quick Answer

**The `input_reshape` layer receives:**
- **Input shape:** `(batch_size, 150)` = `(1, 150)` for single prediction
- **Output shape:** `(batch_size, 30, 5)` = `(1, 30, 5)`

Where:
- `150` = `30 timesteps × 5 features` (flattened)
- `30` = WINDOW_SIZE (number of historical timesteps)
- `5` = OHLCV features (Open, High, Low, Close, Volume)

---

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Environment Observation                                 │
│ Shape: (30, 5)                                                  │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────┐   │
│ │ Timestep 0:  [open_0,  high_0,  low_0,  close_0,  vol_0] │   │
│ │ Timestep 1:  [open_1,  high_1,  low_1,  close_1,  vol_1] │   │
│ │ Timestep 2:  [open_2,  high_2,  low_2,  close_2,  vol_2] │   │
│ │     ...                                                   │   │
│ │ Timestep 29: [open_29, high_29, low_29, close_29, vol_29]│   │
│ └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ .flatten()
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Flatten to 1D Array                                     │
│ Shape: (150,)                                                   │
│                                                                 │
│ [o0, h0, l0, c0, v0, o1, h1, l1, c1, v1, ..., o29, h29, l29,   │
│  c29, v29]                                                      │
│                                                                 │
│ Total: 30 × 5 = 150 values                                     │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ np.expand_dims(axis=0)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Add Batch Dimension                                     │
│ Shape: (1, 150)                                                 │
│                                                                 │
│ [[o0, h0, l0, c0, v0, o1, h1, l1, c1, v1, ..., o29, h29, l29,  │
│   c29, v29]]                                                    │
│                                                                 │
│ ⚠️  THIS IS THE INPUT TO THE MODEL!                            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ Enter Model
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: input_reshape Layer (Inside Model)                      │
│                                                                 │
│ Input:  (1, 150)                                                │
│ Reshape: (1, 30, 5)                                             │
│ Output: (1, 30, 5)                                              │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────┐   │
│ │ Batch 0:                                                 │   │
│ │   Timestep 0:  [open_0,  high_0,  low_0,  close_0,  v0] │   │
│ │   Timestep 1:  [open_1,  high_1,  low_1,  close_1,  v1] │   │
│ │   Timestep 2:  [open_2,  high_2,  low_2,  close_2,  v2] │   │
│ │       ...                                                │   │
│ │   Timestep 29: [open_29, high_29, low_29, close_29, v29]│   │
│ └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ LSTM can now process as sequence
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: LSTM Layer 1                                            │
│ Input:  (1, 30, 5)   - 30 timesteps, 5 features each           │
│ Output: (1, 30, 64)  - 30 timesteps, 64 hidden features        │
│                                                                 │
│ Processes each timestep sequentially, maintaining memory        │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: LSTM Layer 2                                            │
│ Input:  (1, 30, 64)  - 30 timesteps, 64 hidden features        │
│ Output: (1, 32)      - Final state only, 32 hidden features    │
│                                                                 │
│ Returns only the final hidden state (last timestep)            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: Dense Layer                                             │
│ Input:  (1, 32)                                                 │
│ Output: (1, 64)                                                 │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 8: Output Layer (Q-values)                                 │
│ Input:  (1, 64)                                                 │
│ Output: (1, 3)                                                  │
│                                                                 │
│ [Q_hold, Q_buy, Q_sell]                                         │
│ Example: [0.123, 0.456, -0.789]                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Breakdown

### Input to `input_reshape` Layer

```python
# Shape: (batch_size, window_size * num_features)
# Example with batch_size=1, window_size=30, num_features=5:

input_data = np.array([
    [o0, h0, l0, c0, v0,    # Timestep 0
     o1, h1, l1, c1, v1,    # Timestep 1
     o2, h2, l2, c2, v2,    # Timestep 2
     ...
     o29, h29, l29, c29, v29]  # Timestep 29
])

# Shape: (1, 150)
```

### Output from `input_reshape` Layer

```python
# Shape: (batch_size, window_size, num_features)
# Example with batch_size=1, window_size=30, num_features=5:

reshaped_data = np.array([
    [
        [o0,  h0,  l0,  c0,  v0],   # Timestep 0
        [o1,  h1,  l1,  c1,  v1],   # Timestep 1
        [o2,  h2,  l2,  c2,  v2],   # Timestep 2
        ...
        [o29, h29, l29, c29, v29]   # Timestep 29
    ]
])

# Shape: (1, 30, 5)
```

---

## Why This Shape?

### LSTM Requirements

LSTM layers expect 3D input:
1. **Dimension 0:** Batch size (how many samples)
2. **Dimension 1:** Timesteps (sequence length)
3. **Dimension 2:** Features (data per timestep)

### Our Configuration

- **Batch size:** 1 (single prediction at a time)
- **Timesteps:** 30 (WINDOW_SIZE = 30)
- **Features:** 5 (OHLCV)

### The Reshape Operation

```python
# Before reshape: (1, 150)
# Think of it as: 1 sample with 150 values in a flat array

# After reshape: (1, 30, 5)
# Think of it as: 1 sample with 30 timesteps, each having 5 features

# The reshape operation groups every 5 consecutive values
# into one timestep:
[o0, h0, l0, c0, v0, o1, h1, l1, c1, v1, ...]
     ↓
[[o0, h0, l0, c0, v0],
 [o1, h1, l1, c1, v1],
 ...]
```

---

## Code Examples

### In Environment (`crypto_env.py`)

```python
def _get_observation(self):
    # Extract OHLCV data
    raw_data = self.df[['open', 'high', 'low', 'close', 'volume']].values[start_idx:end_idx]
    # Shape: (30, 5)
    
    # Normalize and flatten
    flattened = normalized_data.flatten()
    # Shape: (150,)
    
    return flattened.astype(np.float32)
```

### In Backtester (`evaluate_nobitex_dqn.py`)

```python
# Extract last WINDOW_SIZE OHLCV data
state = features[i-WINDOW_SIZE:i, :]  # Shape: (30, 5)
state = state.flatten()                # Shape: (150,)
state = np.expand_dims(state, axis=0)  # Shape: (1, 150)

# Now feed to model
q_values = model.predict(state)        # Input: (1, 150), Output: (1, 3)
```

### In Model (`q_network.py`)

```python
model = tf.keras.Sequential([
    # This layer receives (1, 150) and outputs (1, 30, 5)
    layers.Reshape((window_size, num_features), input_shape=input_shape),
    
    # LSTM can now process the sequence
    layers.LSTM(64, return_sequences=True),
    ...
])
```

---

## Summary Table

| Stage | Shape | Description |
|-------|-------|-------------|
| Environment observation | `(30, 5)` | 30 timesteps × 5 OHLCV features |
| After flatten | `(150,)` | All 150 values in 1D array |
| After batch dimension | `(1, 150)` | **INPUT TO MODEL** |
| After input_reshape | `(1, 30, 5)` | Ready for LSTM processing |
| After LSTM1 | `(1, 30, 64)` | 30 timesteps × 64 hidden features |
| After LSTM2 | `(1, 32)` | Final state only |
| After Dense | `(1, 64)` | Feature extraction |
| Final output | `(1, 3)` | Q-values for 3 actions |

---

## Key Takeaway

**The `input_reshape` layer transforms:**
- **From:** `(1, 150)` - Flat array of all OHLCV values
- **To:** `(1, 30, 5)` - Structured sequence for LSTM

This allows the LSTM to understand that the data represents **30 sequential timesteps**, each with **5 features**, rather than just 150 random numbers!
