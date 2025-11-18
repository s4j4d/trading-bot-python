"""
Q-Network model creation and management functionality.

This module contains the neural network model creation logic for the DQN trading bot.
"""

import tensorflow as tf
from tensorflow.keras import layers
import os

# Import LSTM-specific configuration constants
from config.constants import (
    LSTM_LEARNING_RATE,
    LSTM_GRADIENT_CLIP_NORM,
    LSTM_LAYER_1_UNITS,
    LSTM_LAYER_2_UNITS,
    LSTM_DROPOUT_RATE,
    LSTM_RECURRENT_DROPOUT,
    LSTM_DENSE_UNITS
)


def _is_lstm_compatible(model):
    """
    Validates that a loaded model is compatible with LSTM-based processing.
    
    This function checks if a loaded model has the expected architecture for LSTM processing,
    including proper input/output shapes and layer types.
    
    Args:
        model (tf.keras.Model): The loaded model to validate
        
    Returns:
        bool: True if model is LSTM-compatible, False otherwise
    """
    try:
        # Check if model has the expected number of outputs (3 for Hold/Buy/Sell)
        if model.output_shape[-1] != 3:
            print(f"Model output shape mismatch: expected 3 actions, got {model.output_shape[-1]}")
            return False
        
        # Check if model contains LSTM layers (indicates new architecture)
        has_lstm = any('lstm' in layer.name.lower() for layer in model.layers)
        
        # Check if model can process the expected input shape
        # Try a test prediction with dummy data to verify compatibility
        try:
            # Create dummy input matching expected shape (batch_size=1, window_size=50)
            dummy_input = tf.zeros((1, 50))
            _ = model(dummy_input, training=False)
            print(f"Model compatibility check passed. LSTM layers present: {has_lstm}")
            return True
        except Exception as e:
            print(f"Model failed input compatibility test: {e}")
            return False
            
    except Exception as e:
        print(f"Error during model compatibility check: {e}")
        return False


def _validate_input_shape(input_shape):
    """
    Validates that input shape is compatible with LSTM processing.
    
    Performs comprehensive validation to ensure the input shape can be properly
    processed by LSTM layers, including checks for dimensionality, size constraints,
    and memory requirements.
    
    Args:
        input_shape (tuple): Shape of the input observations (window_size, num_features)
        
    Raises:
        ValueError: If input shape is not compatible with LSTM processing
        TypeError: If input_shape is not a tuple or contains non-integer values
    """
    # Type validation
    if not isinstance(input_shape, tuple):
        raise TypeError(f"Input shape must be a tuple, got {type(input_shape).__name__}")
    
    # Dimensionality validation for LSTM compatibility
    if len(input_shape) != 2:
        raise ValueError(
            f"Expected 2D input shape for LSTM processing (window_size, num_features), got {len(input_shape)}D shape: {input_shape}. "
            f"LSTM models expect observations as (timesteps, features)."
        )
    
    # Extract and validate dimensions
    try:
        window_size = int(input_shape[0])
        num_features = int(input_shape[1])
    except (ValueError, TypeError) as e:
        raise TypeError(f"Input shape dimensions must be integers, got {input_shape}")
    
    # Validate number of features
    if num_features != 5:
        raise ValueError(
            f"Expected 5 features (OHLCV), got {num_features}. "
            f"Input shape should be (window_size, 5)"
        )
    
    # Window size range validation for LSTM temporal learning
    if window_size < 1:
        raise ValueError(f"Window size must be positive, got {window_size}")
    
    if window_size < 10:
        raise ValueError(
            f"Window size {window_size} is too small for effective LSTM temporal learning. "
            f"Minimum recommended window size is 10 for capturing meaningful sequential patterns. "
            f"Current WINDOW_SIZE configuration should be at least 10."
        )
    
    if window_size > 200:
        raise ValueError(
            f"Window size {window_size} is too large for efficient LSTM processing. "
            f"Maximum recommended window size is 200 to prevent vanishing gradients and memory issues. "
            f"Consider reducing WINDOW_SIZE in configuration."
        )
    
    # Optimal range validation for LSTM
    if window_size < 20:
        print(f"WARNING: Window size {window_size} is below optimal range (20-100) for LSTM temporal learning.")
    elif window_size > 100:
        print(f"WARNING: Window size {window_size} is above optimal range (20-100) and may impact training efficiency.")
    
    # Memory usage estimation and warning
    estimated_memory_mb = (window_size * num_features * 4 * 128) / (1024 * 1024)  # Rough estimate for batch processing
    if estimated_memory_mb > 100:
        print(f"WARNING: Large input size ({window_size}, {num_features}) may require significant memory (~{estimated_memory_mb:.1f}MB per batch)")
    
    print(f"Input shape validation passed: {input_shape} (window_size={window_size}, features={num_features})")


def _create_lstm_model(input_shape, action_size):
    """
    Creates a new LSTM-based Q-Network model with gradient clipping and memory optimization.
    
    This function creates an LSTM model specifically designed for sequential market data processing
    with built-in safeguards against exploding gradients and memory issues.
    
    Args:
        input_shape (tuple): Shape of the input observations (window_size, num_features)
        action_size (int): Number of possible actions
        
    Returns:
        tf.keras.Model: Compiled LSTM-based model with gradient clipping
        
    Raises:
        ValueError: If model creation fails due to invalid parameters
        tf.errors.ResourceExhaustedError: If insufficient memory for model creation
    """
    # Extract window size and number of features from input shape
    # Input shape is (window_size, num_features) where num_features = 5 (OHLCV)
    window_size = input_shape[0]
    num_features = input_shape[1]
    
    print(f"Model input configuration: window_size={window_size}, num_features={num_features}")
    
    try:
        # Validate LSTM configuration parameters before model creation
        _validate_lstm_configuration()
        
        # Estimate memory requirements and adjust architecture if needed
        estimated_params = _estimate_model_parameters(window_size, action_size, num_features)
        print(f"Estimated model parameters: {estimated_params:,}")
        
        # Create model with memory-efficient LSTM configuration using constants
        # No reshape needed! Input is already in correct shape (window_size, num_features)
        model = tf.keras.Sequential([
            
            # Input layer to define shape explicitly (no reshape needed!)
            layers.Input(shape=input_shape, name='input_layer'),
            
            # First LSTM layer: configurable units with return_sequences=True to pass full sequence to next layer
            # Use recurrent_dropout for additional regularization in LSTM cells
            layers.LSTM(LSTM_LAYER_1_UNITS, 
                       return_sequences=True, 
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       recurrent_dropout=LSTM_RECURRENT_DROPOUT,
                       name='lstm_layer_1'),
            
            # Dropout layer between LSTM layers for regularization
            layers.Dropout(LSTM_DROPOUT_RATE, name='dropout_1'),
            
            # Second LSTM layer: configurable units with return_sequences=False to output final hidden state
            layers.LSTM(LSTM_LAYER_2_UNITS, 
                       return_sequences=False, 
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       recurrent_dropout=LSTM_RECURRENT_DROPOUT,
                       name='lstm_layer_2'),
            
            # Dropout layer after LSTM processing
            layers.Dropout(LSTM_DROPOUT_RATE, name='dropout_2'),
            
            # Dense layer for feature extraction: configurable units with ReLU activation
            layers.Dense(LSTM_DENSE_UNITS, activation='relu', name='dense_features'),
            
            # Output layer: 3 units with linear activation for Q-values (Hold, Buy, Sell)
            layers.Dense(action_size, activation='linear', name='q_values_output')
        ])
        
        # Create optimizer with gradient clipping to prevent exploding gradients
        # Gradient clipping is crucial for LSTM training stability
        # Use configurable learning rate and clipping norm optimized for LSTM
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LSTM_LEARNING_RATE,
            clipnorm=LSTM_GRADIENT_CLIP_NORM  # Clip gradients by norm to prevent exploding gradients
        )
        
        # Compile the model with gradient clipping optimizer
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae']  # Add mean absolute error for additional monitoring
        )
        
        print(f"LSTM model created successfully with gradient clipping (clipnorm={LSTM_GRADIENT_CLIP_NORM})")
        print(f"LSTM configuration: Layer1={LSTM_LAYER_1_UNITS} units, Layer2={LSTM_LAYER_2_UNITS} units, LR={LSTM_LEARNING_RATE}")
        return model
        
    except tf.errors.ResourceExhaustedError as e:
        raise tf.errors.ResourceExhaustedError(
            f"Insufficient memory to create LSTM model with window_size={window_size}. "
            f"Try reducing the window size or batch size. Original error: {e}"
        )
    except ValueError as e:
        raise ValueError(f"Invalid parameters for LSTM model creation: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error during LSTM model creation: {type(e).__name__}: {e}")


def _validate_lstm_configuration():
    """
    Validates that all LSTM configuration parameters are properly set for temporal learning.
    
    Raises:
        ValueError: If any configuration parameter is invalid for LSTM training
    """
    # Validate learning rate
    if LSTM_LEARNING_RATE <= 0 or LSTM_LEARNING_RATE > 0.01:
        raise ValueError(
            f"LSTM_LEARNING_RATE ({LSTM_LEARNING_RATE}) should be between 0 and 0.01 for stable training. "
            f"Recommended range: 0.0001 to 0.001"
        )
    
    # Validate gradient clipping
    if LSTM_GRADIENT_CLIP_NORM <= 0 or LSTM_GRADIENT_CLIP_NORM > 10:
        raise ValueError(
            f"LSTM_GRADIENT_CLIP_NORM ({LSTM_GRADIENT_CLIP_NORM}) should be between 0 and 10. "
            f"Recommended range: 0.5 to 2.0"
        )
    
    # Validate LSTM layer sizes
    if LSTM_LAYER_1_UNITS < 16 or LSTM_LAYER_1_UNITS > 256:
        raise ValueError(
            f"LSTM_LAYER_1_UNITS ({LSTM_LAYER_1_UNITS}) should be between 16 and 256. "
            f"Recommended range: 32 to 128"
        )
    
    if LSTM_LAYER_2_UNITS < 8 or LSTM_LAYER_2_UNITS > 128:
        raise ValueError(
            f"LSTM_LAYER_2_UNITS ({LSTM_LAYER_2_UNITS}) should be between 8 and 128. "
            f"Recommended range: 16 to 64"
        )
    
    # Validate dropout rates
    if LSTM_DROPOUT_RATE < 0 or LSTM_DROPOUT_RATE >= 0.8:
        raise ValueError(
            f"LSTM_DROPOUT_RATE ({LSTM_DROPOUT_RATE}) should be between 0 and 0.8. "
            f"Recommended range: 0.1 to 0.3"
        )
    
    if LSTM_RECURRENT_DROPOUT < 0 or LSTM_RECURRENT_DROPOUT >= 0.5:
        raise ValueError(
            f"LSTM_RECURRENT_DROPOUT ({LSTM_RECURRENT_DROPOUT}) should be between 0 and 0.5. "
            f"Recommended range: 0.05 to 0.2"
        )
    
    # Validate dense layer size
    if LSTM_DENSE_UNITS < 16 or LSTM_DENSE_UNITS > 256:
        raise ValueError(
            f"LSTM_DENSE_UNITS ({LSTM_DENSE_UNITS}) should be between 16 and 256. "
            f"Recommended range: 32 to 128"
        )
    
    print("LSTM configuration validation passed - all parameters are within recommended ranges")


def _estimate_model_parameters(window_size, action_size, num_features=5):
    """
    Estimates the number of parameters in the LSTM model for memory planning.
    Uses configurable LSTM layer sizes for accurate parameter estimation.
    
    Args:
        window_size (int): Size of the input sequence
        action_size (int): Number of output actions
        num_features (int): Number of input features per timestep (default: 5 for OHLCV)
        
    Returns:
        int: Estimated number of model parameters
    """
    # LSTM layer 1: configurable units with input size = num_features (OHLCV)
    # 4 gates * units * (input_size + hidden_size + bias)
    lstm1_params = 4 * LSTM_LAYER_1_UNITS * (num_features + LSTM_LAYER_1_UNITS + 1)
    
    # LSTM layer 2: configurable units with input size from layer 1
    lstm2_params = 4 * LSTM_LAYER_2_UNITS * (LSTM_LAYER_1_UNITS + LSTM_LAYER_2_UNITS + 1)
    
    # Dense layer: configurable units with input size from LSTM layer 2
    dense_params = LSTM_DENSE_UNITS * (LSTM_LAYER_2_UNITS + 1)
    
    # Output layer: action_size units with input size from dense layer
    output_params = action_size * (LSTM_DENSE_UNITS + 1)
    
    total_params = lstm1_params + lstm2_params + dense_params + output_params
    return total_params


def build_q_network(input_shape, action_size):
    """
    Builds an LSTM-based Q-Network for Deep Q-Learning with backward compatibility.
    
    This function implements a robust model loading strategy that:
    1. Attempts to load existing model files first (backward compatibility)
    2. Validates loaded models for architecture compatibility
    3. Falls back to creating new LSTM model when loading fails
    4. Supports both old Dense-based and new LSTM-based model formats
    
    The model uses LSTM layers to process sequential market data and capture temporal dependencies.
    The model takes market data observations as input and outputs Q-values for each possible action.
    
    Args:
        input_shape (tuple): Shape of the input observations from the environment
                           Expected to be (WINDOW_SIZE,) for flat observation window
        action_size (int): Number of possible actions the agent can take
                          For trading: 3 actions (Hold=0, Buy=1, Sell=2)
        
    Returns:
        tf.keras.Model: Compiled Q-Network model ready for Q-learning
                       - Input: Market observation data (price history sequence)
                       - Output: Q-values for each action (Hold, Buy, Sell)
                       
    Raises:
        ValueError: If input_shape is not compatible with model processing
        Exception: If model creation fails after all loading attempts
    """
    # Validate input shape before proceeding
    _validate_input_shape(input_shape)
    
    # Define model file paths in order of preference
    # Try most recent models first, then fall back to older ones

    model_paths = ['final_parallel_trading_model.keras']
    
    # Attempt to load existing models with backward compatibility
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"Attempting to load existing model from {model_path}...")
                model = tf.keras.models.load_model(model_path)
                
                # Validate that the loaded model is compatible with our requirements
                if _is_lstm_compatible(model):
                    print(f"Successfully loaded compatible model from {model_path}")
                    print("Loaded model summary:")
                    model.summary()
                    return model
                else:
                    print(f"Model from {model_path} is not compatible with current requirements")
                    print("Will attempt to load next model or create new LSTM model")
                    continue
                    
            except tf.errors.InvalidArgumentError as e:
                print(f"Model loading failed - Invalid model format in {model_path}: {e}")
                print("This may indicate a corrupted model file or incompatible TensorFlow version")
                continue
            except tf.errors.NotFoundError as e:
                print(f"Model loading failed - Missing model components in {model_path}: {e}")
                print("The model file may be incomplete or corrupted")
                continue
            except ValueError as e:
                print(f"Model loading failed - Value error with {model_path}: {e}")
                print("This may indicate incompatible model architecture or configuration")
                continue
            except OSError as e:
                print(f"Model loading failed - File system error with {model_path}: {e}")
                print("Check file permissions and disk space")
                continue
            except Exception as e:
                print(f"Model loading failed - Unexpected error with {model_path}: {type(e).__name__}: {e}")
                print("Will attempt to load next model or create new LSTM model")
                continue
    
    # If no existing compatible model found, create new LSTM-based model
    print("No compatible existing model found. Creating new LSTM-based Q-Network...")
    
    try:
        model = _create_lstm_model(input_shape, action_size)
        print("New LSTM-based model created successfully!")
        print("New model summary:")
        model.summary()
        return model
        
    except Exception as e:
        print(f"Failed to create new LSTM model: {e}")
        raise Exception(f"Unable to create Q-Network model: {e}")
    
    # This should never be reached, but included for completeness
    raise Exception("Unexpected error in model creation process")