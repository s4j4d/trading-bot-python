"""
Cryptocurrency trading environment for reinforcement learning.

This module contains the CryptoTradingEnv class which implements a Gymnasium
environment for simulating cryptocurrency trading with DQN.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import warnings

from config.constants import (
    WINDOW_SIZE,
    INITIAL_BALANCE,
    FEE_RATE,
    SLIPPAGE_FACTOR,
    MAX_STEPS_PER_EPISODE
)


class CryptoTradingEnv(gym.Env):
    """
    A Gymnasium environment for simulating cryptocurrency trading with DQN.
    
    This environment simulates a trading scenario where an agent can:
    - Hold (do nothing)
    - Buy cryptocurrency 
    - Sell cryptocurrency
    
    The agent receives rewards based on portfolio performance.
    """
    # Metadata required by Gymnasium - defines rendering capabilities
    metadata = {'render_modes': [], 'render_fps': 1}

    def __init__(self, df,
                 window_size=WINDOW_SIZE,
                 initial_balance=INITIAL_BALANCE,
                 fee_rate=FEE_RATE,
                 slippage_factor=SLIPPAGE_FACTOR,
                 max_steps=MAX_STEPS_PER_EPISODE):
        """
        Initialize the cryptocurrency trading environment.
        
        Args:
            df: DataFrame containing market data with 'close' price column
            window_size: Number of historical price points to include in observation
            initial_balance: Starting cash balance for trading
            fee_rate: Transaction fee as a percentage (e.g., 0.001 = 0.1%)
            slippage_factor: Price slippage factor for realistic trading simulation
            max_steps: Maximum number of steps per episode
        """
        super().__init__()

        # Validate input data
        if df.empty or len(df) <= window_size:
             raise ValueError("DataFrame is empty or too short for the window size.")

        # Ensure df has enough data for max_steps + window_size
        if len(df) < max_steps + window_size:
             warnings.warn(f"DataFrame length ({len(df)}) is less than max_steps ({max_steps}) + window_size ({window_size}). Episode might end early.", RuntimeWarning)
             # Adjust max_steps if df is too short to avoid index errors later
             self.max_steps = max(1, len(df) - window_size) # Ensure at least 1 step is possible
        else:
             self.max_steps = max_steps

        # Store market data - copy to avoid modifying original
        self.df = df.copy()
        
        # Environment configuration parameters
        self.window_size = window_size          # Number of price points in observation window
        self.initial_balance = initial_balance  # Starting cash amount
        self.fee_rate = fee_rate               # Transaction fee percentage
        self.slippage_factor = slippage_factor # Price impact factor for trades

        # Define action and observation spaces for Gymnasium
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: window of historical prices
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.window_size,), dtype=np.float32
        )

        # Trading state variables (initialized here, set in reset())
        self.current_step = 0                    # Current position in the data
        self.balance = 0.0                       # Current cash balance
        self.holdings = 0.0                      # Current cryptocurrency holdings
        self.initial_portfolio_value = 0.0       # Portfolio value at episode start
        self.previous_portfolio_value = 0.0      # Portfolio value from previous step

    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            tuple: (initial_observation, info_dict)
        """
        super().reset(seed=seed)

        # Start index needs to account for window size - we need historical data
        # Start at window_size-1 so we have enough history for the first observation
        self.current_step = self.window_size - 1
        
        # Reset trading state to initial conditions
        self.balance = self.initial_balance           # Reset cash to starting amount
        self.holdings = 0.0                          # Start with no cryptocurrency
        self.initial_portfolio_value = self.initial_balance  # Record starting portfolio value
        self.previous_portfolio_value = self.initial_balance # Initialize previous value

        # Get initial observation and info
        state = self._get_observation()
        info = self._get_info()

        # Validate that our observation matches the expected shape
        if state.shape != self.observation_space.shape:
             raise ValueError(f"Reset state shape {state.shape} does not match observation space shape {self.observation_space.shape}")

        return state, info

    def step(self, action):
        # --- Boundary Check ---
        # Check if the *next* step would exceed the data frame length or max_steps
        if self.current_step >= len(self.df) - 1 or (self.current_step - (self.window_size - 1)) >= self.max_steps :
             # If already at or beyond the limit, return previous state and indicate done
             current_price = self._get_current_price() # Price at the terminal step
             current_portfolio_value = self.balance + self.holdings * current_price
             reward = 0 # No change in value at the terminal step itself
             done = True
             truncated = (self.current_step - (self.window_size - 1)) >= self.max_steps and self.current_step < len(self.df) -1

             state = self._get_observation() # Observation at the terminal step
             info = self._get_info()
             info["final_portfolio_value"] = current_portfolio_value # Add final value info

             # Ensure state matches the defined space shape even at termination
             if state.shape != self.observation_space.shape:
                  raise ValueError(f"Terminal step state shape {state.shape} does not match observation space shape {self.observation_space.shape}")

             return state, reward, done, truncated, info

        # --- Proceed with normal step ---
        current_price = self._get_current_price()
        value_before_action = self.balance + self.holdings * current_price

        # --- Execute action ---
        if action == 1 and self.balance > 1e-6:  # Buy
            effective_price = current_price * (1 + self.slippage_factor)
            buy_amount_in_cash = self.balance
            fee = buy_amount_in_cash * self.fee_rate
            cash_after_fee = buy_amount_in_cash - fee
            if effective_price > 1e-9:
                crypto_bought = cash_after_fee / effective_price
                self.holdings += crypto_bought
                self.balance = 0.0
            else:
                 warnings.warn(f"Step {self.current_step}: Buy skipped due to near-zero effective price.", RuntimeWarning)

        elif action == 2 and self.holdings > 1e-8:  # Sell
            effective_price = current_price * (1 - self.slippage_factor)
            sell_amount_in_crypto = self.holdings
            cash_received_before_fee = sell_amount_in_crypto * effective_price
            fee = cash_received_before_fee * self.fee_rate
            cash_received_after_fee = cash_received_before_fee - fee
            self.balance += cash_received_after_fee
            self.holdings = 0.0

        # --- Update state and check termination ---
        self.current_step += 1 # Increment step *after* calculations based on current_step

        # Check termination conditions *after* incrementing step
        terminated = self.current_step >= len(self.df) -1 # Ran out of data
        truncated = (self.current_step - (self.window_size - 1)) >= self.max_steps # Reached step limit

        done = terminated or truncated

        # --- Calculate reward ---
        # Use the price corresponding to the *new* current_step
        price_for_value_calc = self._get_current_price()
        current_portfolio_value = self.balance + self.holdings * price_for_value_calc
        reward = current_portfolio_value - self.previous_portfolio_value
        self.previous_portfolio_value = current_portfolio_value

        # --- Get next observation and info ---
        next_state = self._get_observation() # Observation for the *new* current_step
        info = self._get_info()
        if done:
             info["final_portfolio_value"] = current_portfolio_value # Add final value info

        # Ensure state matches the defined space shape
        if next_state.shape != self.observation_space.shape:
             raise ValueError(f"Step state shape {next_state.shape} does not match observation space shape {self.observation_space.shape}")

        return next_state, reward, terminated, truncated, info # Return terminated and truncated separately

    def _get_current_price(self):
        """
        Get the current market price at the current step.
        
        Returns:
            float: Current closing price of the cryptocurrency
        """
        # Ensure index is within bounds to prevent IndexError
        # If current_step exceeds data length, use the last available price
        safe_step = min(self.current_step, len(self.df) - 1)
        return float(self.df.iloc[safe_step]['close'])

    def _get_observation(self):
         # Ensure index is within bounds for observation window
         safe_step = min(self.current_step, len(self.df) - 1)
         end_idx = safe_step + 1
         start_idx = max(0, end_idx - self.window_size)
         raw_prices = self.df['close'].values[start_idx:end_idx]

         # Handle cases where data starts before window size is reached
         if len(raw_prices) < self.window_size:
             padding_value = self.df['close'].iloc[0] if len(self.df) > 0 else 0
             padding_value = float(padding_value)
             padding = np.full(self.window_size - len(raw_prices), padding_value, dtype=np.float64)
             raw_prices = raw_prices.astype(np.float64)
             raw_prices = np.concatenate((padding, raw_prices))

         # --- Normalization ---
         last_price = raw_prices[-1]
         if last_price > 1e-8:
             normalized_prices = raw_prices / last_price
         else:
             normalized_prices = np.zeros_like(raw_prices)
             # Avoid warning spamming at the very beginning if initial prices are zero
             if safe_step >= self.window_size:
                  warnings.warn(f"Near-zero price encountered at step {safe_step}, observation may be unreliable.", RuntimeWarning)

         if len(normalized_prices) != self.window_size:
              # This should theoretically not happen with the padding, but good failsafe
              raise ValueError(f"Observation shape incorrect at step {self.current_step}. Expected {self.window_size}, got {len(normalized_prices)}")

         return normalized_prices.astype(np.float32)

    def _get_info(self):
        """
        Get additional information about the current state.
        
        Returns:
            dict: Dictionary containing current trading state information
        """
        return {
            "step": self.current_step,                    # Current step number in episode
            "balance": self.balance,                      # Current cash balance
            "holdings": self.holdings,                    # Current cryptocurrency holdings
            "current_price": self._get_current_price(),   # Current market price
            "portfolio_value": self.previous_portfolio_value  # Total portfolio value (cash + holdings value)
        }