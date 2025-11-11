"""
Main training functionality for the cryptocurrency trading bot.
"""

import tensorflow as tf
import numpy as np
import time
import gc
from collections import deque
import random
from gymnasium.vector import AsyncVectorEnv

# Optional import for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not available. Memory monitoring will be limited.")

# Import from other modules
from data import load_data_from_json
from environment import CryptoTradingEnv
from model import build_q_network
from training.utils import linear_decay
from config.constants import (
    WINDOW_SIZE,
    INITIAL_BALANCE,
    MAX_STEPS_PER_EPISODE,
    NUM_ENVS,
    TOTAL_STEPS,
    REPLAY_BUFFER_SIZE,
    BATCH_SIZE,
    MIN_REPLAY_SIZE,
    TARGET_UPDATE_FREQ_STEPS,
    LEARNING_STEPS,
    GAMMA,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY_TIMESTEPS,
    JSON_FILE_PATH
)


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


def _adjust_batch_size_for_memory(initial_batch_size, memory_threshold_gb=0.5):
    """
    Dynamically adjusts batch size based on available memory.
    
    Args:
        initial_batch_size (int): Starting batch size
        memory_threshold_gb (float): Minimum available memory threshold in GB
        
    Returns:
        int: Adjusted batch size
    """
    memory_stats = _monitor_memory_usage()
    available_memory = memory_stats.get('ram_available_gb', 4.0)
    
    if available_memory < memory_threshold_gb:
        # Reduce batch size if memory is low
        reduction_factor = max(0.25, available_memory / memory_threshold_gb)
        adjusted_batch_size = max(16, int(initial_batch_size * reduction_factor))
        print(f"Memory low ({available_memory:.2f}GB available). Reducing batch size from {initial_batch_size} to {adjusted_batch_size}")
        return adjusted_batch_size
    
    return initial_batch_size


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


def train_trading_bot():
    """
    Main training function for the cryptocurrency trading bot using Deep Q-Network (DQN).
    
    This function implements a complete DQN training pipeline with the following components:
    - Parallel environment execution for faster data collection
    - Experience replay buffer for stable learning
    - Target network for stable Q-value targets
    - Epsilon-greedy exploration strategy with linear decay
    - Periodic logging and model saving
    
    The training process:
    1. Loads historical market data from JSON file
    2. Creates multiple parallel trading environments
    3. Builds main and target Q-networks
    4. Collects experiences through environment interaction
    5. Trains the network using experience replay
    6. Updates target network periodically
    7. Saves the final trained model
    
    Raises:
        FileNotFoundError: If the market data file cannot be found
        ValueError: If the data or configuration is invalid
        Exception: For any other unexpected errors during training
    """
    try:
        # Configure GPU settings for optimal performance
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s). Configuring for training...")
            try:
                # Enable memory growth to prevent TensorFlow from allocating all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth enabled.")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPU found. Training will use CPU (this will be slower).")
            # Optimize CPU performance
            tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all CPU cores
            tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all CPU cores
            print("CPU optimization enabled - using all available cores.")
        
        # Print device placement for verification
        print("Available devices:")
        for device in tf.config.list_physical_devices():
            print(f"  {device}")
        print()
        # Load historical cryptocurrency market data from JSON file
        # df_crypto contains OHLCV (Open, High, Low, Close, Volume) data
        df_crypto = load_data_from_json(JSON_FILE_PATH)

        # --- Create Vectorized Environment ---
        print(f"\n--- Creating {NUM_ENVS} parallel environments ---")
        
        # Create multiple parallel trading environments for faster data collection
        # Each environment runs independently with the same market data
        # Lambda functions ensure each environment gets its own copy of parameters
        env_fns = [lambda: CryptoTradingEnv(
            df_crypto, 
            window_size=WINDOW_SIZE, 
            initial_balance=INITIAL_BALANCE, 
            max_steps=MAX_STEPS_PER_EPISODE
        ) for _ in range(NUM_ENVS)]
        
        # AsyncVectorEnv runs environments in separate processes for true parallelism
        vec_env = AsyncVectorEnv(env_fns)

        # Extract environment specifications for neural network architecture
        single_observation_space = vec_env.single_observation_space  # Shape of one environment's observations
        single_action_space = vec_env.single_action_space            # Action space of one environment
        state_shape = single_observation_space.shape                 # Input shape for neural network
        action_size = single_action_space.n                          # Number of possible actions (Hold, Buy, Sell)
        
        # Verify WINDOW_SIZE configuration matches environment observation shape
        if state_shape[0] != WINDOW_SIZE:
            raise ValueError(
                f"Environment observation shape ({state_shape[0]}) does not match WINDOW_SIZE ({WINDOW_SIZE}). "
                f"Please ensure WINDOW_SIZE in constants.py matches the environment configuration."
            )
        
        print(f"LSTM sequence configuration verified:")
        print(f"  - Observation shape: {state_shape} (matches WINDOW_SIZE: {WINDOW_SIZE})")
        print(f"  - Action size: {action_size}")
        print(f"  - Sequence length for temporal learning: {WINDOW_SIZE} time steps")

        # --- Build Networks --------------------------------------------------------------------------------------------------------------------------------
        # Main Q-network: learns to predict Q-values for state-action pairs
        # This network is updated frequently during training
        q_network = build_q_network(state_shape, action_size)
        
        # Target Q-network: provides stable targets for Q-value updates
        # This network is updated less frequently to prevent training instability
        target_network = build_q_network(state_shape, action_size)
        target_network.set_weights(q_network.get_weights())  # Initialize with same weights

        # Experience replay buffer: stores past experiences for training
        # Deque with maxlen automatically removes old experiences when full
        replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

        # Initialize adaptive batch size for memory management
        # BATCH_SIZE is now optimized for LSTM training (reduced from 128 to 64)
        current_batch_size = _adjust_batch_size_for_memory(BATCH_SIZE)
        memory_error_count = 0
        max_memory_errors = 3  # Maximum number of memory errors before stopping

        print("\n--- Starting Training ---")
        print(f"LSTM-optimized configuration:")
        print(f"  - Batch size: {current_batch_size} (adjusted from {BATCH_SIZE})")
        print(f"  - Window size: {WINDOW_SIZE} (optimized for temporal learning)")
        print(f"  - Target update frequency: {TARGET_UPDATE_FREQ_STEPS} steps")
        print(f"  - Learning frequency: every {LEARNING_STEPS} steps")
        
        # Log initial memory usage
        initial_memory = _monitor_memory_usage()
        print(f"Initial memory usage - RAM: {initial_memory['ram_used_gb']:.2f}GB ({initial_memory['ram_percent']:.1f}%)")
        if 'gpu_current_mb' in initial_memory:
            print(f"Initial GPU memory: {initial_memory['gpu_current_mb']:.1f}MB")
        
        start_time = time.time()

        # --- Initialization for Vector Env ---------------------------------------------------------------------------------------------------------------------------
        # Reset all environments to get initial states
        states, infos = vec_env.reset()  # Returns initial observations for all environments
        
        # Training progress tracking variables
        total_done_steps = 0                                              # Total steps across all environments
        total_episodes_finished = 0                                  # Count of completed episodes
        episode_rewards = np.zeros(NUM_ENVS)                        # Current episode reward for each environment
        finished_episode_rewards = deque(maxlen=100)                # Rolling window of last 100 episode rewards
        losses = deque(maxlen=100)                                   # Rolling window of recent training losses

        while total_done_steps < TOTAL_STEPS:

            # --- Epsilon-greedy action selection for all environments ---
            # Calculate current exploration probability (epsilon) using linear decay
            # Starts high for exploration, decreases over time for exploitation
            # epsilon = linear_decay(total_done_steps, EPSILON_DECAY_TIMESTEPS, EPSILON_START, EPSILON_END)
            epsilon = 0.2

            if np.random.rand() < epsilon:
                # Exploration: Take random actions to discover new strategies
                # vec_env.action_space.sample() returns random actions for all environments
                actions = vec_env.action_space.sample()
            else:
                # Exploitation: Use learned Q-network to select best actions
                # Convert states to tensor for neural network processing
                states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
                # Get Q-values for all possible actions in current states
                q_values = q_network(states_tensor, training=False)
                # Select actions with highest Q-values (argmax along action dimension)
                actions = tf.argmax(q_values, axis=1).numpy()
            # --- Step all environments ---
            # Execute selected actions in all environments simultaneously
            # Returns: next observations, rewards, termination flags, truncation flags, info dicts
            next_states, rewards, terminations, truncations, infos = vec_env.step(actions)

            # --- Store experiences from all environments ---
            # Combine termination and truncation flags to determine which episodes ended
            # terminations: episode ended naturally (goal reached, failure, etc.)
            # truncations: episode ended due to time limit or other constraints
            dones = np.logical_or(terminations, truncations)

            # Process experiences from each environment
            for i in range(NUM_ENVS):
                # Store experience tuple (s, a, r, s', done) in replay buffer
                # This forms the basis for Q-learning updates
                replay_buffer.append((
                    states[i],      # Current state
                    actions[i],     # Action taken
                    rewards[i],     # Reward received
                    next_states[i], # Resulting next state
                    dones[i]        # Whether episode ended
                ))

                # Accumulate rewards for current episode in this environment
                episode_rewards[i] += rewards[i]

                # Handle episode completion
                if dones[i]:
                    total_episodes_finished += 1
                    # Record the total reward for this completed episode
                    finished_episode_rewards.append(episode_rewards[i])
                    # Reset reward accumulator for next episode in this environment
                    episode_rewards[i] = 0

                    # Check for final info provided by vec env wrapper
                    # if "final_info" in infos:
                    #     final_info = infos["final_info"][i]
                    #     if final_info is not None and "final_portfolio_value" in final_info:
                    #          print(f"  Final Portfolio Value (Env {i}): {final_info['final_portfolio_value']:.2f}")


            # Update the current states for the next iteration
            # next_states become the current states for the next training loop
            states = next_states
            
            # Increment total step counter by number of environments
            # Each environment contributes one step per iteration
            total_done_steps += NUM_ENVS

            # --- Training Step ----------------------------------------------------------------------------------------------------------------------------------------
            # Only train if we have enough experiences and it's time to learn
            # MIN_REPLAY_SIZE ensures diverse experiences before training starts
            # LEARNING_STEPS controls how often we perform training updates
            if len(replay_buffer) >= MIN_REPLAY_SIZE and total_done_steps % LEARNING_STEPS == 0:
                
                try:
                    # Sample a random batch of experiences from replay buffer
                    # Use current_batch_size which may be adjusted for memory constraints
                    batch_size_to_use = min(current_batch_size, len(replay_buffer))
                    batch = random.sample(replay_buffer, batch_size_to_use)

                    # Unpack batch into separate arrays for efficient processing
                    # Each element contains: (state, action, reward, next_state, done)
                    states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb = map(np.array, zip(*batch))

                    # Convert numpy arrays to TensorFlow tensors for neural network processing
                    states_tensor = tf.convert_to_tensor(states_mb, dtype=tf.float32)      # Current states
                    actions_tensor = tf.convert_to_tensor(actions_mb, dtype=tf.int32)      # Actions taken
                    rewards_tensor = tf.convert_to_tensor(rewards_mb, dtype=tf.float32)    # Rewards received
                    next_states_tensor = tf.convert_to_tensor(next_states_mb, dtype=tf.float32)  # Next states
                    # Convert boolean done flags to float for mathematical operations
                    dones_tensor = tf.convert_to_tensor(dones_mb, dtype=tf.float32)        # Episode termination flags

                    # Calculate target Q-values using the target network
                    # Target network provides stable Q-value estimates for next states
                    next_q_values_target = target_network(next_states_tensor, training=False)

                    # Apply Bellman equation to calculate target Q-values
                    # Q*(s,a) = r + γ * max(Q*(s',a')) if not terminal, else r
                    target_max_q = tf.reduce_max(next_q_values_target, axis=1)  # Best Q-value for next state
                    # Multiply by (1.0 - dones_tensor) to zero out future rewards for terminal states
                    targets = rewards_tensor + GAMMA * target_max_q * (1.0 - dones_tensor)

                    # Use GradientTape to track operations for automatic differentiation
                    with tf.GradientTape() as tape:
                        # Get Q-values for current states from the main network
                        current_q_values = q_network(states_tensor, training=True)

                        # Extract Q-values for the specific actions that were taken
                        # Create 2D indices: [[batch_idx, action], ...] for gather_nd operation
                        action_indices = tf.stack([
                            tf.range(len(actions_tensor), dtype=tf.int32),  # Batch indices [0, 1, 2, ...]
                            actions_tensor                                   # Action indices for each batch item
                        ], axis=1)
                        # Get Q-values for the actions actually taken
                        action_q_values = tf.gather_nd(current_q_values, action_indices)

                        # Calculate loss between predicted Q-values and target Q-values
                        # This measures how well our network predicts the true Q-values
                        loss_value = q_network.compiled_loss(targets, action_q_values)

                    # Calculate gradients of loss with respect to network parameters
                    # Gradients are automatically clipped by the optimizer (configured in model creation)
                    gradients = tape.gradient(loss_value, q_network.trainable_variables)

                    # Check for None gradients which can indicate numerical issues
                    if any(grad is None for grad in gradients):
                        print("WARNING: None gradients detected. Skipping this training step.")
                        continue

                    # Apply gradients to update network weights using the optimizer with gradient clipping
                    q_network.optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

                    # Record loss for monitoring training progress
                    losses.append(loss_value.numpy())
                    
                    # Reset memory error count on successful training step
                    memory_error_count = 0

                except tf.errors.ResourceExhaustedError as e:
                    # Handle GPU/CPU memory exhaustion
                    memory_error_count += 1
                    print(f"Memory exhausted during training (error {memory_error_count}/{max_memory_errors}): {e}")
                    
                    if memory_error_count >= max_memory_errors:
                        print("Maximum memory errors reached. Stopping training to prevent system instability.")
                        break
                    
                    # Reduce batch size and clean up memory
                    current_batch_size = _handle_memory_error(current_batch_size)
                    
                    # Skip this training step
                    continue
                    
                except tf.errors.InvalidArgumentError as e:
                    print(f"Invalid argument error during training: {e}")
                    print("This may indicate a problem with input data shapes or model architecture.")
                    # Continue training but log the error
                    continue
                    
                except Exception as e:
                    print(f"Unexpected error during training step: {type(e).__name__}: {e}")
                    # Continue training but log the error
                    continue

            # --- Update Target Network Periodically ---
            # Copy weights from main network to target network at regular intervals
            # This provides stable learning targets and prevents training instability
            if total_done_steps % TARGET_UPDATE_FREQ_STEPS == 0 and total_done_steps > 0:
                target_network.set_weights(q_network.get_weights())

            # --- Logging ---
            # Print training progress at regular intervals
            # Log approximately every MAX_STEPS_PER_EPISODE interactions per environment
            if total_done_steps % (MAX_STEPS_PER_EPISODE * NUM_ENVS) == 0 and total_done_steps > 0:
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Calculate training performance metrics
                steps_per_second = total_done_steps / elapsed_time if elapsed_time > 0 else 0  # Training speed
                avg_reward = np.mean(finished_episode_rewards) if finished_episode_rewards else 0.0  # Performance
                avg_loss = np.mean(losses) if losses else 0.0  # Learning progress

                # Get current memory usage for monitoring
                current_memory = _monitor_memory_usage()

                # Print comprehensive training status with memory monitoring
                print(f"Total Steps: {total_done_steps:8d}/{TOTAL_STEPS} | "
                      f"Episodes Finished: {total_episodes_finished:6d} | "
                      f"Avg Reward (Last 100): {avg_reward:10.2f} | "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"Epsilon: {epsilon:.3f} | "
                      f"Buffer Size: {len(replay_buffer):6d} | "
                      f"Batch Size: {current_batch_size} | "
                      f"SPS: {steps_per_second:.0f}")
                
                # Print memory usage information
                if 'psutil_unavailable' in current_memory:
                    print("Memory monitoring: psutil not available", end="")
                else:
                    print(f"Memory - RAM: {current_memory['ram_used_gb']:.2f}GB ({current_memory['ram_percent']:.1f}%) | "
                          f"Available: {current_memory['ram_available_gb']:.2f}GB", end="")
                
                if 'gpu_current_mb' in current_memory:
                    print(f" | GPU: {current_memory['gpu_current_mb']:.1f}MB")
                else:
                    print()
                
                # Warning if memory usage is high (only if psutil is available)
                if PSUTIL_AVAILABLE and current_memory['ram_percent'] > 85:
                    print("WARNING: High memory usage detected. Consider reducing batch size or buffer size.")
                
                print()  # Add blank line for readability

        print("\n--- Training Finished ---")
        end_time = time.time()
        print(f"Total training time: {(end_time - start_time)/3600:.2f} hours")

        # --- Cleanup ---
        print("Closing vector environment...")
        vec_env.close()

        # --- Save Final Model ---
        # Optionally save the final model
        try:
            save_path = "final_parallel_trading_model.keras"
            q_network.save(save_path)
            # files.download("final_parallel_trading_model.keras")
            print(f"Saved final model to {save_path}")
            # Example of saving weights only:
            # q_network.save_weights("final_parallel_trading_model.weights.h5")
        except Exception as e:
             print(f"Error saving final model: {e}")

    except FileNotFoundError as e:
        print(f"\n--- Data Loading Error ---")
        print(e)
        print("Please ensure the JSON_FILE_PATH variable points to the correct file.")
    except ValueError as e: # Catch specific ValueErrors from Env or data loading/env creation
        print(f"\n--- Configuration or Data Error ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        import traceback
        traceback.print_exc()