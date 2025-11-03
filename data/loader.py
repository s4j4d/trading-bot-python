"""
Data loading module for the cryptocurrency trading bot.

This module provides functionality to load and process historical OHLCV data
from JSON files for use in the trading environment.
"""

import json
import os
import warnings
import pandas as pd


def load_data_from_json(json_file_path):
    """
    Loads historical OHLCV data from a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file containing market data.
                             Expected format: {"t": [...], "o": [...], "h": [...], 
                             "l": [...], "c": [...], "v": [...]}
    
    Returns:
        pd.DataFrame: DataFrame with timestamp index and columns for open, high, 
                     low, close, and volume data. All numeric columns are converted 
                     to float type.
    
    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
        ValueError: If the JSON data is malformed or missing required keys.
        json.JSONDecodeError: If the file contains invalid JSON.
    
    Example:
        >>> df = load_data_from_json('market_data.json')
        >>> print(df.head())
                            open    high     low   close    volume
        timestamp                                                
        2023-01-01 00:00:00  100.0   105.0    95.0   102.0  1000.0
    """
    print(f"Attempting to load data from: {json_file_path}")
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found at: {json_file_path}")

    try:
        # Load JSON data
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded JSON data.")

        # Validate data structure
        required_keys = ["t", "o", "h", "l", "c", "v"]
        if not isinstance(data, dict):
            raise ValueError("JSON data is not a dictionary (object). Expected API-like structure.")

        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"JSON data missing required keys: {missing_keys}")

        if not all(isinstance(data[key], list) for key in required_keys):
            raise ValueError(f"One or more required keys in JSON do not map to lists.")

        # Check and handle different list lengths
        list_lengths = {key: len(data[key]) for key in required_keys}
        min_len = min(list_lengths.values()) if list_lengths else 0
        if len(set(list_lengths.values())) > 1:
            warnings.warn(
                f"Data lists have different lengths: {list_lengths}. "
                f"Truncating to shortest length: {min_len}", 
                RuntimeWarning
            )
            for key in required_keys:
                if key in data and isinstance(data[key], list):
                    data[key] = data[key][:min_len]

        if min_len == 0:
            raise ValueError("JSON data lists became empty after processing.")

        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["t"], unit='s', errors='coerce'),
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "close": data["c"],
            "volume": data["v"]
        })
        df.set_index("timestamp", inplace=True)

        # Convert numeric columns and handle invalid values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Clean data by removing NaN values
        initial_rows = len(df)
        df.dropna(inplace=True)
        if len(df) < initial_rows:
            print(f"Dropped {initial_rows - len(df)} rows with NaN or invalid values during processing.")

        if df.empty:
            raise ValueError(f"DataFrame became empty after cleaning data loaded from {json_file_path}.")

        # Ensure all data is float type and sorted by timestamp
        df = df.astype(float)
        df.sort_index(inplace=True)

        print(f"Successfully processed {len(df)} data points from {json_file_path}.")
        print("skipping the first 1000 used in the previous training")
        
        return df[1000:]

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {json_file_path}: {e}")
        raise
    except Exception as e:
        print(f"An error occurred during data loading or processing from {json_file_path}: {e}")
        raise