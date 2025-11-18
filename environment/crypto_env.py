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
        
        # Observation space: window of historical OHLCV data + position indicator
        # Shape: (window_size, 6) - 5 OHLCV features + 1 position indicator
        # Position indicator: 0 = holding cash, 1 = holding crypto
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.window_size, 6), dtype=np.float32
        )

        # Trading state variables (initialized here, set in reset())
        self.current_step = 0                    # Current position in the data
        self.balance = 0.0                       # Current cash balance
        self.holdings = 0.0                      # Current cryptocurrency holdings
        self.initial_portfolio_value = 0.0       # Portfolio value at episode start
        self.previous_portfolio_value = 0.0      # Portfolio value from previous step
        self.last_buy_price = 0.0                # Track last buy price for profit calculation
        self.last_sell_price = 0.0               # Track last sell price

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
        self.last_buy_price = 0.0                    # Reset last buy price
        self.last_sell_price = 0.0                   # Reset last sell price

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
             current_price = self._get_current_closing_price() # Price at the terminal step
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
        current_price = self._get_current_closing_price()
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
                self.last_buy_price = effective_price  # Track buy price
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
            self.last_sell_price = effective_price  # Track sell price

        # --- Update state and check termination ---
        self.current_step += 1 # Increment step *after* calculations based on current_step

        # Check termination conditions *after* incrementing step
        terminated = self.current_step >= len(self.df) -1 # Ran out of data
        truncated = (self.current_step - (self.window_size - 1)) >= self.max_steps # Reached step limit

        done = terminated or truncated

        # --- Calculate reward --------------------------------------------------------------------------------------------------------------------------------------
        # Use the price corresponding to the *new* current_step
        price_for_value_calc = self._get_current_closing_price()
        current_portfolio_value = self.balance + self.holdings * price_for_value_calc
        
        # Base reward: change in portfolio value (normalized by portfolio size)
        portfolio_change = current_portfolio_value - self.previous_portfolio_value
        reward = portfolio_change / self.initial_balance * 10000  # Scale up for better learning
        
        # AGGRESSIVE: Penalize holding positions (encourages active trading)
        if action == 0:  # Hold action
            if self.holdings > 1e-8:  # Holding crypto
                # Penalty scales with portfolio value
                reward -= 500  # Fixed penalty for holding crypto
            elif self.balance > 1e-6:  # Holding cash
                reward -= 300  # Fixed penalty for holding cash
        
        # STRONG BONUS: Reward successful trades
        if action == 1 and self.balance < 1e-6:  # Successfully bought (now all-in crypto)
            reward += 200  # Bonus for executing buy
            
        elif action == 2 and self.holdings < 1e-8:  # Successfully sold (now all cash)
            reward += 200  # Bonus for executing sell
        
        # EXTRA BONUS: Reward good trade timing (buy low, sell high)
        if action == 1 and self.last_sell_price > 1e-9:  # Just bought after selling
            if current_price < self.last_sell_price:
                price_improvement = (self.last_sell_price - current_price) / self.last_sell_price
                reward += 5000 * price_improvement  # Big bonus for buying the dip
        
        elif action == 2 and self.last_buy_price > 1e-9:  # Just sold after buying
            if current_price > self.last_buy_price:
                profit_margin = (current_price - self.last_buy_price) / self.last_buy_price
                reward += 5000 * profit_margin  # Big bonus for selling high
        
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

    def _get_current_closing_price(self):
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
         """
         Get observation with all OHLCV features + position indicator.
         
         Returns:
             np.ndarray: Array of normalized OHLCV data + position
                        Shape: (window_size, 6) containing [open, high, low, close, volume, position]
         """
         # Ensure index is within bounds for observation window
         safe_step = min(self.current_step, len(self.df) - 1)
         end_idx = safe_step + 1
         start_idx = max(0, end_idx - self.window_size)
         
         # Extract all OHLCV features
         raw_data = self.df[['open', 'high', 'low', 'close', 'volume']].values[start_idx:end_idx]

         # Handle cases where data starts before window size is reached
         if len(raw_data) < self.window_size:
             # Use first row for padding
             padding_row = self.df[['open', 'high', 'low', 'close', 'volume']].iloc[0].values if len(self.df) > 0 else np.zeros(5)
             padding_row = padding_row.astype(np.float64)
             padding = np.tile(padding_row, (self.window_size - len(raw_data), 1))
             raw_data = raw_data.astype(np.float64)
             raw_data = np.vstack((padding, raw_data))

         # --- Normalization ---
         # Normalize prices by last close price, volume by its own max
         last_close = raw_data[-1, 3]  # Last close price (index 3)
         
         if last_close > 1e-8:
             # Normalize OHLC by last close price
             normalized_data = raw_data.copy()
             normalized_data[:, :4] = raw_data[:, :4] / last_close  # Normalize open, high, low, close
             
             # Normalize volume separately (by max volume in window to keep scale reasonable)
             max_volume = np.max(raw_data[:, 4])
             if max_volume > 1e-8:
                 normalized_data[:, 4] = raw_data[:, 4] / max_volume
             else:
                 normalized_data[:, 4] = 0.0
         else:
             normalized_data = np.zeros_like(raw_data)
             # Avoid warning spamming at the very beginning if initial prices are zero
             if safe_step >= self.window_size:
                  warnings.warn(f"Near-zero price encountered at step {safe_step}, observation may be unreliable.", RuntimeWarning)

         # Add position indicator column (0 = cash, 1 = crypto)
         position_indicator = 1.0 if self.holdings > 1e-8 else 0.0
         position_column = np.full((self.window_size, 1), position_indicator, dtype=np.float32)
         
         # Concatenate position indicator to observation
         observation = np.hstack([normalized_data, position_column])

         # Validate shape
         if observation.shape != (self.window_size, 6):
              raise ValueError(f"Observation shape incorrect at step {self.current_step}. Expected ({self.window_size}, 6), got {observation.shape}")

         return observation.astype(np.float32)

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
            "current_price": self._get_current_closing_price(),   # Current market price
            "portfolio_value": self.previous_portfolio_value  # Total portfolio value (cash + holdings value)
        }