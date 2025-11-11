"""
Model module for the cryptocurrency trading bot.

This module contains neural network model definitions and utilities for
Deep Q-Network (DQN) based trading. It provides functions to build and
configure Q-networks for reinforcement learning.

The module provides:
- Q-network architecture definition
- Model building and configuration utilities
- Pre-trained model loading functionality

Usage:
    from model import build_q_network
    
    # Build a Q-network for trading
    q_net = build_q_network(input_shape=(30,), action_size=3)
"""

from .q_network import build_q_network

# Export public API
__all__ = ['build_q_network']