#!/usr/bin/env python3
"""
Main entry point for the cryptocurrency trading bot.

This script orchestrates the entire training process by importing all necessary
modules and calling the main training function.
"""

from training.trainer import train_trading_bot


def main():
    """
    Main entry point that orchestrates the training process.
    
    This function serves as the primary coordinator for the entire training pipeline:
    1. Initializes the training environment
    2. Calls the main training function
    3. Handles errors and user interruptions gracefully
    4. Provides user feedback throughout the process
    
    The function wraps the training process in error handling to ensure:
    - Clean shutdown on user interruption (Ctrl+C)
    - Proper error reporting for debugging
    - Clear status messages for user feedback
    """
    print("Starting Cryptocurrency Trading Bot Training...")
    print("=" * 50)
    
    try:
        # Execute the main training pipeline
        # This will load data, create environments, train the model, and save results
        train_trading_bot()
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        # Handle user interruption (Ctrl+C) gracefully
        print("\nTraining interrupted by user.")
    except Exception as e:
        # Handle any other errors that occur during training
        print(f"\nTraining failed with error: {e}")
        raise  # Re-raise the exception for debugging purposes


if __name__ == "__main__":
    main()