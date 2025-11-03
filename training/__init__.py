"""
Training module for the cryptocurrency trading bot.

This module contains the complete training pipeline for Deep Q-Network (DQN)
based cryptocurrency trading. It includes the main training loop, utility
functions, and supporting components for reinforcement learning.

The module provides:
- Main training orchestration (train_trading_bot)
- Utility functions for training (linear_decay)
- Experience replay and target network management
- Parallel environment handling and data collection

Components:
- trainer.py: Main training loop and DQN implementation
- utils.py: Helper functions for training (epsilon decay, etc.)

Usage:
    from training import train_trading_bot
    
    # Start the complete training process
    train_trading_bot()
"""

from .trainer import train_trading_bot
from .utils import linear_decay

# Export public API
__all__ = ['train_trading_bot', 'linear_decay']