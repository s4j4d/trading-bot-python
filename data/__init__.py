"""
Data module for the cryptocurrency trading bot.

This module provides functionality for loading and processing market data.
"""

from .loader import load_data_from_json

__all__ = ['load_data_from_json']