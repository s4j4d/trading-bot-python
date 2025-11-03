"""
Data fetching utility for the cryptocurrency trading bot.

This script fetches real-time historical market data from the Nobitex API
and saves it in the format expected by the trading bot's data loader.
"""

import time
import json
import requests


def fetch_real_df():
    """
    Fetches historical OHLCV data from Nobitex API and saves it to a JSON file.
    
    This function retrieves market data for a specific cryptocurrency symbol
    and time range, then saves it in the format expected by the trading bot.
    
    The function performs the following steps:
    1. Configures API request parameters (symbol, resolution, time range)
    2. Makes HTTP request to Nobitex API
    3. Validates the API response
    4. Saves the data to 'market_data.json' file
    
    API Parameters:
        symbol (str): Cryptocurrency trading pair symbol ("WIRT")
        resolution (str): Time interval for candlestick data ("5" = 5 minutes)
        from_time (int): Start timestamp for data retrieval (Unix timestamp)
        to_time (int): End timestamp for data retrieval (Unix timestamp)
    
    Data Format:
        The API returns OHLCV data in the following structure:
        {
            "t": [timestamps],     # Unix timestamps
            "o": [open_prices],    # Opening prices
            "h": [high_prices],    # Highest prices
            "l": [low_prices],     # Lowest prices
            "c": [close_prices],   # Closing prices
            "v": [volumes],        # Trading volumes
            "s": "ok"              # Status indicator
        }
    
    Raises:
        Exception: If the API request fails or returns an error status
        requests.RequestException: If there are network connectivity issues
    
    Note:
        This function is designed for data collection and testing purposes.
        The main training pipeline uses the saved JSON file for consistency.
    """
    # Trading pair symbol - WIRT (Iranian Rial to Tether)
    symbol = "WIRT"
    
    # Time resolution for candlestick data (5 = 5-minute intervals)
    # Other options: 1, 5, 15, 30, 60, 240, 1D, 1W, 1M
    resolution = "5"
    
    # Calculate time range for data retrieval
    # End time: 1500 hours ago (to avoid most recent incomplete data)
    to_time = int(time.time())
    
    # Start time: 2000 hours before end time (total span: 2000 hours of data)
    from_time = to_time - ((30*24-3)* 3600)
    
    # Nobitex API endpoint for historical market data
    url = "https://apiv2.nobitex.ir/market/udf/history"
    
    # API request parameters
    params = {
        "symbol": symbol,        # Trading pair to fetch
        "resolution": resolution, # Time interval for candles
        "from": from_time,       # Start timestamp
        "to": to_time           # End timestamp
    }
    
    # Make HTTP GET request to the API
    response = requests.get(url, params=params)
    data = response.json()
    
    # Validate API response status
    if data["s"] != "ok":
        raise Exception(f"API request failed: {data}")
    
    # Remove status field as it's not needed for training data
    data.pop("s")
    
    # Save the market data to JSON file for use by the trading bot
    with open('market_data.json', 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Data saved to market_data.json")
    print(f"Fetched {len(data.get('t', []))} data points")
    print(f"Time range: {from_time} to {to_time}")
    
    # Note: DataFrame creation is handled by the data loader module
    # This keeps the data fetching and processing concerns separated
    # The main training pipeline uses load_data_from_json() for consistent processing


# Execute the data fetching when script is run directly
if __name__ == "__main__":
    fetch_real_df()