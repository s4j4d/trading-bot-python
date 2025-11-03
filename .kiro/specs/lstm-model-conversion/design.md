# Design Document: LSTM Model Conversion

## Overview

This design document outlines the conversion of the existing feedforward neural network to an LSTM-based architecture for the cryptocurrency trading bot. The current model uses a simple Dense layer architecture that processes the observation window as independent features. The new LSTM architecture will treat the observation window as a temporal sequence, enabling the model to learn from sequential patterns and temporal dependencies in price data.

The conversion maintains backward compatibility with the existing training pipeline while enhancing the model's ability to capture market momentum, trends, and temporal patterns that are crucial for effective trading decisions.

## Architecture

### Current Architecture Analysis

The existing Q-network uses a feedforward architecture:
- Input: Flattened observation window of shape `(WINDOW_SIZE,)` where `WINDOW_SIZE = 50`
- Hidden layers: Dense(128) → Dropout(0.2) → Dense(128) → Dropout(0.2) → Dense(64)
- Output: Dense(3) for Q-values (Hold, Buy, Sell actions)
- The model treats each price point in the window as an independent feature

### New LSTM Architecture

The new architecture will process the observation window as a temporal sequence:

```
Input Layer: (batch_size, WINDOW_SIZE, 1)
    ↓
LSTM Layer 1: 64 units, return_sequences=True
    ↓
Dropout: 0.2
    ↓
LSTM Layer 2: 32 units, return_sequences=False
    ↓
Dropout: 0.2
    ↓
Dense Layer: 64 units, ReLU activation
    ↓
Output Layer: 3 units, Linear activation (Q-values)
```

### Key Architectural Changes

1. **Input Reshaping**: Transform input from `(batch_size, WINDOW_SIZE)` to `(batch_size, WINDOW_SIZE, 1)`
2. **LSTM Layers**: Replace Dense layers with LSTM layers to process sequential data
3. **Temporal Processing**: Enable the model to learn from the order and relationships between consecutive price points
4. **Memory Cells**: LSTM cells will maintain internal state to remember important patterns across the sequence

## Components and Interfaces

### Modified Q-Network Function

The `build_q_network()` function will be updated while maintaining the same interface:

```python
def build_q_network(input_shape, action_size):
    """
    Builds an LSTM-based Q-Network for Deep Q-Learning.
    
    Args:
        input_shape (tuple): Shape of the input observations (WINDOW_SIZE,)
        action_size (int): Number of possible actions (3 for Hold/Buy/Sell)
        
    Returns:
        tf.keras.Model: Compiled LSTM-based Keras model
    """
```

### Input Processing Layer

A new input processing component will handle the reshaping:
- **Purpose**: Convert flat observation window to 3D tensor for LSTM processing
- **Input**: `(batch_size, WINDOW_SIZE)` - flattened price sequence
- **Output**: `(batch_size, WINDOW_SIZE, 1)` - reshaped for LSTM
- **Implementation**: Using Keras Reshape layer or Lambda layer

### LSTM Processing Layers

Two LSTM layers with different configurations:
- **LSTM Layer 1**: 64 units, `return_sequences=True` to pass full sequence to next layer
- **LSTM Layer 2**: 32 units, `return_sequences=False` to output final hidden state
- **Dropout**: Applied between LSTM layers for regularization

### Output Processing Layer

Dense layers for final Q-value prediction:
- **Dense Layer**: 64 units with ReLU activation for feature extraction
- **Output Layer**: 3 units with linear activation for Q-values

## Data Models

### Input Data Structure

```python
# Current format (maintained for compatibility)
observation_shape = (WINDOW_SIZE,)  # (50,)
observation_data = np.array([price1, price2, ..., price50])

# Internal LSTM processing format
lstm_input_shape = (WINDOW_SIZE, 1)  # (50, 1)
lstm_input_data = observation_data.reshape(-1, WINDOW_SIZE, 1)
```

### Model State Management

The LSTM model will maintain internal state between sequence elements but will reset state between episodes:
- **Stateful**: False (state resets between batches)
- **Sequence Length**: Fixed at WINDOW_SIZE (50 time steps)
- **Features per Time Step**: 1 (normalized price value)

### Backward Compatibility

The model will handle both old and new model formats:
- **Loading**: Attempt to load existing `.keras` files first
- **Fallback**: Create new LSTM model if loading fails
- **Interface**: Maintain same function signature and return type

## Error Handling

### Model Loading Errors

```python
try:
    model = tf.keras.models.load_model(model_path)
    # Verify model architecture compatibility
    if not _is_lstm_compatible(model):
        raise ValueError("Loaded model is not LSTM-compatible")
except Exception as e:
    print(f"Failed to load model: {e}")
    # Fall back to creating new LSTM model
```

### Input Shape Validation

```python
def _validate_input_shape(input_shape):
    """Validate that input shape is compatible with LSTM processing."""
    if len(input_shape) != 1:
        raise ValueError(f"Expected 1D input shape, got {input_shape}")
    if input_shape[0] != WINDOW_SIZE:
        raise ValueError(f"Expected window size {WINDOW_SIZE}, got {input_shape[0]}")
```

### Memory Management

- **Gradient Clipping**: Implement gradient clipping to prevent exploding gradients
- **Memory Monitoring**: Add checks for GPU memory usage during training
- **Batch Size Adjustment**: Automatically reduce batch size if memory errors occur

## Testing Strategy

### Unit Tests

1. **Model Creation Tests**
   - Verify LSTM model architecture is created correctly
   - Test input/output shapes match expectations
   - Validate model compilation succeeds

2. **Input Processing Tests**
   - Test reshaping from flat to 3D tensor
   - Verify temporal order is preserved
   - Check batch processing works correctly

3. **Backward Compatibility Tests**
   - Test loading of existing non-LSTM models fails gracefully
   - Verify new LSTM models can be saved and loaded
   - Check interface compatibility with existing code

### Integration Tests

1. **Training Pipeline Integration**
   - Test LSTM model works with existing DQN training loop
   - Verify experience replay mechanism functions correctly
   - Check target network updates work properly

2. **Environment Compatibility**
   - Test model processes environment observations correctly
   - Verify action selection produces valid outputs
   - Check model performance in trading environment

### Performance Tests

1. **Training Speed Comparison**
   - Benchmark LSTM vs Dense model training time
   - Monitor memory usage during training
   - Test batch processing efficiency

2. **Inference Performance**
   - Measure prediction latency for single observations
   - Test batch prediction performance
   - Verify real-time trading compatibility

### Validation Tests

1. **Temporal Learning Verification**
   - Create synthetic sequential data with known patterns
   - Verify LSTM learns temporal dependencies
   - Compare against feedforward model on sequential tasks

2. **Trading Performance Tests**
   - Backtest on historical data
   - Compare trading performance metrics
   - Analyze decision patterns for temporal awareness