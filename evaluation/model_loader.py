"""
Model Loading Module
===================
Handles loading trained DQN models.
"""

import tensorflow as tf


def load_trained_model(model_path: str = "final_parallel_trading_model.keras") -> tf.keras.Model:
    """
    Load the trained DQN model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded TensorFlow model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    try:
        print(f"Loading trained model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return model
        
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Please ensure the model file exists or train a new model first.")
        raise
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("The model file may be corrupted or incompatible.")
        raise
