"""
Configuration constants for the trading bot.

This module contains all the constants and configuration parameters
used throughout the trading bot system. These values control the behavior
of the trading environment, neural network training, and data processing.
"""

# --- Environment Constants ---
# Number of historical price points to include in each observation
# Optimized for LSTM temporal learning - provides sufficient sequence length
# for capturing short to medium-term market patterns while maintaining efficiency
# REDUCED for high-frequency trading: shorter memory for faster reactions
WINDOW_SIZE = 30  # Shorter window for day trading: focus on recent price action

# INITIAL_CRYPTO_BALANCE: float = 0.0

# Starting cash balance for each trading episode (in currency units)
# Set to 10 million for realistic portfolio simulation
# INITIAL_FIAT_BALANCE = 10000000.0
INITIAL_BALANCE = 10000000.0


# Transaction fee rate as a decimal (0.003 = 0.3%)
# Applied to each buy/sell transaction to simulate real trading costs
FEE_RATE = 0.003

# Price slippage factor for market impact simulation (0.0005 = 0.05%)
# Accounts for the price movement caused by large trades
SLIPPAGE_FACTOR = 0.0005

# Maximum number of trading steps allowed per episode
# Prevents episodes from running indefinitely and controls training time
MAX_STEPS_PER_EPISODE = 1500

# --- Training Parameters ---
# Number of parallel trading environments to run simultaneously
# More environments provide diverse experiences but require more memory
# INCREASED: Using more environments to utilize available CPU/memory resources
NUM_ENVS = 500

EPISODES = 1500

# Total number of training steps across all environments
# Higher values lead to better learning but longer training time
# Episode is a training data
TOTAL_STEPS = EPISODES * MAX_STEPS_PER_EPISODE

# Size of the experience replay buffer for storing past experiences
# Larger buffers provide more diverse training data but use more memory
REPLAY_BUFFER_SIZE = 20000

# Number of experiences to sample from replay buffer for each training step
# Optimized for LSTM training: smaller batches help with memory management
# and gradient stability for recurrent networks
# INCREASED: Larger batches for better GPU utilization and faster convergence
BATCH_SIZE = 1024  # Increased from 512 for better hardware utilization

# Minimum number of experiences required before starting training
# Ensures sufficient data diversity before learning begins
MIN_REPLAY_SIZE = 5000

# Frequency (in steps) for updating the target network
# More frequent updates for LSTM to adapt to temporal pattern changes
# while maintaining training stability
# INCREASED: Less frequent updates to reduce computational overhead
TARGET_UPDATE_FREQ_STEPS = 2000  # Reduced frequency for faster training

# Number of learning steps to perform for every N environment steps
# Adjusted for LSTM: more frequent learning to capture temporal patterns
# while preventing overfitting on sequential data
# REDUCED: Less frequent learning for faster training (more data collection per learning step)
LEARNING_STEPS = 8  # Reduced frequency for faster training

# Discount factor for future rewards in Q-learning
# Values closer to 1 make the agent consider long-term rewards more heavily
# LOWERED for high-frequency trading: focus on short-term volatility profits
# 0.1-0.3 = very short-term (scalping), 0.3-0.5 = day trading, 0.7-0.85 = swing trading, 0.99 = long-term
GAMMA = 0.7  # Higher for better planning and avoiding bad trades

# --- Epsilon-greedy Exploration Parameters ---
# Starting probability of taking random actions (exploration)
# Higher values encourage more initial exploration
EPSILON_START = 0.5

# Final probability of taking random actions after decay
# INCREASED for active trading: maintain some exploration to find trading opportunities
EPSILON_END = 0.10  # Lower exploration for more consistent trading decisions

# Number of timesteps over which to decay epsilon from start to end
# Decay over 75% of total training time, then maintain minimum exploration
EPSILON_DECAY_TIMESTEPS = int(TOTAL_STEPS * 0.75)

# --- LSTM Model Configuration ---
# Learning rate optimized for LSTM training stability
# Lower rate prevents exploding gradients common in recurrent networks
# INCREASED: Higher learning rate for faster convergence
LSTM_LEARNING_RATE = 0.003  # Increased for faster learning

# Gradient clipping norm to prevent exploding gradients in LSTM
# Critical for stable LSTM training with sequential data
LSTM_GRADIENT_CLIP_NORM = 1.0  # Clips gradients to prevent instability

# LSTM layer configuration
LSTM_LAYER_1_UNITS = 64      # First LSTM layer units for sequence processing
LSTM_LAYER_2_UNITS = 32      # Second LSTM layer units for feature extraction
LSTM_DROPOUT_RATE = 0.2      # Dropout rate for LSTM regularization
LSTM_RECURRENT_DROPOUT = 0.1 # Recurrent dropout for LSTM internal connections

# Dense layer configuration for LSTM model
LSTM_DENSE_UNITS = 64        # Dense layer units after LSTM processing

# --- Data File Path ---
# Path to the JSON file containing historical market data
# Must contain OHLCV (Open, High, Low, Close, Volume) data in API format
JSON_FILE_PATH = 'market_data_2025-11-14-1.json'  # IMPORTANT: Change this path if needed!

# --- Checkpoint Configuration ---
# Number of training steps between checkpoint saves
# Checkpoints allow training to be interrupted and resumed without losing progress
CHECKPOINT_INTERVAL = 50000

# Filename for the checkpoint state file (contains step count and replay buffer)
CHECKPOINT_FILE = 'checkpoint.pkl'

# Filename for the saved model checkpoint
CHECKPOINT_MODEL = 'checkpoint_model.keras'