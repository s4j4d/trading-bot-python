"""
Utility functions for training the cryptocurrency trading bot.
"""

import gc
import tensorflow as tf

# Optional import for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def _monitor_memory_usage():
    """
    Monitors current memory usage and returns memory statistics.
    
    Returns:
        dict: Memory usage statistics including RAM and GPU memory
    """
    memory_stats = {}
    
    # Monitor system RAM usage if psutil is available
    if PSUTIL_AVAILABLE:
        try:
            ram = psutil.virtual_memory()
            memory_stats['ram_used_gb'] = ram.used / (1024**3)
            memory_stats['ram_available_gb'] = ram.available / (1024**3)
            memory_stats['ram_percent'] = ram.percent
        except Exception as e:
            memory_stats['ram_error'] = str(e)
    else:
        # Fallback: provide default values when psutil is not available
        memory_stats['ram_used_gb'] = 0.0
        memory_stats['ram_available_gb'] = 4.0  # Assume 4GB available as fallback
        memory_stats['ram_percent'] = 50.0
        memory_stats['psutil_unavailable'] = True
    
    # Monitor GPU memory if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Get GPU memory info
            gpu_details = tf.config.experimental.get_memory_info('GPU:0')
            memory_stats['gpu_current_mb'] = gpu_details['current'] / (1024**2)
            memory_stats['gpu_peak_mb'] = gpu_details['peak'] / (1024**2)
        except Exception as e:
            memory_stats['gpu_error'] = str(e)
    
    return memory_stats


def _handle_memory_error(current_batch_size):
    """
    Handles memory errors by reducing batch size and cleaning up memory.
    
    Args:
        current_batch_size (int): Current batch size that caused the error
        
    Returns:
        int: New reduced batch size
    """
    # Force garbage collection
    gc.collect()
    
    # Clear TensorFlow session if needed
    try:
        tf.keras.backend.clear_session()
    except:
        pass
    
    # Reduce batch size by half, but keep minimum of 8
    new_batch_size = max(8, current_batch_size // 2)
    print(f"Memory error encountered. Reducing batch size from {current_batch_size} to {new_batch_size}")
    
    return new_batch_size


def linear_decay(step, total_done_steps, start_val, end_val):
    """
    Linearly decays a value from start_val to end_val over a specified number of steps.
    
    This function is commonly used for epsilon decay in reinforcement learning,
    where exploration probability decreases linearly over time to shift from
    exploration to exploitation.
    
    Args:
        step (int): Current step number (0-based)
        total_done_steps (int): Total number of steps over which to perform the decay
        start_val (float): Initial value at step 0
        end_val (float): Final value at step total_done_steps
        
    Returns:
        float: Linearly interpolated value between start_val and end_val
               - At step 0: returns start_val
               - At step total_done_steps: returns end_val
               - At intermediate steps: returns proportionally interpolated value
               - For steps > total_done_steps: returns end_val (clamped)
    
    Example:
        >>> # Decay epsilon from 1.0 to 0.1 over 1000 steps
        >>> epsilon = linear_decay(500, 1000, 1.0, 0.1)  # Returns 0.55
        >>> epsilon = linear_decay(1000, 1000, 1.0, 0.1)  # Returns 0.1
    """
    # Calculate the fraction of decay completed (0.0 to 1.0)
    # min() ensures we don't exceed 1.0 for steps beyond total_done_steps
    fraction = min(1.0, step / total_done_steps)
    
    # Linear interpolation: start + fraction * (end - start)
    return start_val + fraction * (end_val - start_val)



def save_checkpoint(step, q_network, replay_buffer):
    """
    Save training checkpoint to enable resuming training from the current state.
    
    This function saves the Q-network model and essential training state (step count
    and replay buffer) to disk. The checkpoint allows training to be interrupted and
    resumed without losing progress.
    
    Args:
        step (int): Current training step number
        q_network: Main Q-network model to save
        replay_buffer (deque): Experience replay buffer containing past experiences
    
    The function saves two files:
    - Model file (CHECKPOINT_MODEL): TensorFlow Keras model
    - State file (CHECKPOINT_FILE): Pickle file with step count and replay buffer
    
    All errors are caught and logged to prevent training interruption.
    """
    import pickle
    from config.constants import CHECKPOINT_FILE, CHECKPOINT_MODEL
    
    try:
        # Save the Q-network model using TensorFlow's native save format
        q_network.save(CHECKPOINT_MODEL)
        
        # Convert replay buffer from deque to list for pickle serialization
        replay_buffer_list = list(replay_buffer)
        
        # Create checkpoint dictionary with training state
        checkpoint_data = {
            'step': step,
            'replay_buffer': replay_buffer_list
        }
        
        # Save checkpoint data to pickle file
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Log successful checkpoint save
        print(f"Checkpoint saved at step {step}")
        
    except Exception as e:
        # Log error but don't crash training
        print(f"Error saving checkpoint at step {step}: {type(e).__name__}: {e}")


def load_checkpoint():
    """
    Load training checkpoint to resume training from a previously saved state.
    
    This function attempts to load a saved Q-network model and training state
    (step count and replay buffer) from disk. If successful, training can resume
    from where it left off. If no checkpoint exists or loading fails, training
    will start fresh.
    
    Returns:
        tuple: (step, q_network, replay_buffer) if checkpoint exists and loads successfully
               (0, None, None) if no checkpoint exists or loading fails
               - step (int): Training step number from checkpoint
               - q_network: Loaded Q-network model
               - replay_buffer (deque): Restored experience replay buffer
    
    The function expects two files:
    - Model file (CHECKPOINT_MODEL): TensorFlow Keras model
    - State file (CHECKPOINT_FILE): Pickle file with step count and replay buffer
    
    All errors are caught and logged to allow training to start fresh if needed.
    """
    import os
    import pickle
    from collections import deque
    import tensorflow as tf
    from config.constants import CHECKPOINT_FILE, CHECKPOINT_MODEL
    
    try:
        # Check if both checkpoint files exist
        if not os.path.exists(CHECKPOINT_FILE) or not os.path.exists(CHECKPOINT_MODEL):
            print("No checkpoint found. Starting training from scratch.")
            return (0, None, None)
        
        # Load the Q-network model using TensorFlow's load_model
        q_network = tf.keras.models.load_model(CHECKPOINT_MODEL)
        
        # Load checkpoint data from pickle file
        with open(CHECKPOINT_FILE, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Extract step count and replay buffer from checkpoint
        step = checkpoint_data['step']
        replay_buffer_list = checkpoint_data['replay_buffer']
        
        # Convert replay buffer list back to deque with proper maxlen
        from config.constants import REPLAY_BUFFER_SIZE
        replay_buffer = deque(replay_buffer_list, maxlen=REPLAY_BUFFER_SIZE)
        
        # Log successful checkpoint load
        print(f"Checkpoint loaded successfully. Resuming from step {step}")
        print(f"Replay buffer restored with {len(replay_buffer)} experiences")
        
        return (step, q_network, replay_buffer)
        
    except Exception as e:
        # Log error and return default values to start fresh
        print(f"Error loading checkpoint: {type(e).__name__}: {e}")
        print("Starting training from scratch.")
        return (0, None, None)
