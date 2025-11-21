"""
Configuration module for the cryptocurrency trading bot.

This module contains all configuration constants and parameters used throughout
the trading bot system. It centralizes configuration management to ensure
consistency across all components.

The module provides:
- Environment configuration (window size, balance, fees, etc.)
- Training hyperparameters (learning rates, buffer sizes, etc.)
- Exploration parameters (epsilon decay settings)
- Data file paths and other system settings

Usage:
    from config.constants import WINDOW_SIZE, INITIAL_BALANCE
"""

from .constants import *

__all__ = ['*']