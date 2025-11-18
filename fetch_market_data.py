import time
from datetime import datetime 
import json
import requests


def fetch_real_df():
    """
    Fetches historical OHLCV data from Nobitex API and saves it to a JSON file.
    
    API Parameters:
        symbol (str): Cryptocurrency trading pair symbol ("WIRT")
        resolution (str): Time interval for candlestick data ("5" = 5 minutes)
        from_time (int): Start timestamp for data retrieval (Unix timestamp)
        to_time (int): End timestamp for data retrieval (Unix timestamp)
  ***** page (string): There is 500 timestamps per page
    
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
    """
    symbol = "WIRT"
    
    # Time resolution for candlestick data (5 = 5-minute intervals)
    # Other options: 1, 5, 15, 30, 60, 240, 1D, 1W, 1M
    resolution = "5"

    numberOfDaysToCatch = 30
    
    to_time = int(time.time())
    print(f"to time : {to_time}")

    from_time = to_time - ((numberOfDaysToCatch*24)* 3600)
    # from_time = 1762612500
    print(f"from time : {from_time}")

    numberOfDataPoints = ((numberOfDaysToCatch*24)* 3600)/300
    numberOfPages = int(numberOfDataPoints/500)
    
    url = "https://apiv2.nobitex.ir/market/udf/history"

    result = {
        "t":[],
        "o":[],
        "h":[],
        "l":[],
        "c":[],
        "v":[]
    }
    
    for i in range(1,numberOfPages):
        print(i)
        print(range(1,numberOfPages))
        params = {
            "symbol": symbol,        # Trading pair to fetch
            "resolution": resolution, # Time interval for candles
            "from": from_time,       # Start timestamp
            "to": to_time,           # End timestamp
            "page": i
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data["s"] == "no_data":
            break

        if data["s"] != "ok":
            raise Exception(f"API request failed: {data}")

        # Remove status field
        data.pop("s")

        data.get("t", []).extend(result["t"])
        data.get("o", []).extend(result["o"])
        data.get("h", []).extend(result["h"])
        data.get("l", []).extend(result["l"])
        data.get("c", []).extend(result["c"])
        data.get("v", []).extend(result["v"])
        result = data

    
    x = datetime.now()

    fileName = f'market_data_{x.strftime("%G-%m-%d")}.json'

    with open(fileName, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"Data saved to {fileName}")
    print(f"Fetched {len(result.get('t', []))} data points")
    print(f"Time range: {result["t"][0]} to {result["t"][-1]}")
    
    # Note: DataFrame creation is handled by the data loader module
    # This keeps the data fetching and processing concerns separated
    # The main training pipeline uses load_data_from_json() for consistent processing


# Execute the data fetching when script is run directly
if __name__ == "__main__":
    fetch_real_df()