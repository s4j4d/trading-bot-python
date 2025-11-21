"""
Data Processing Module
=====================
Fetches and prepares market data for model input.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from evaluation.nobitex_api import NobitexAPI
from config.constants import WINDOW_SIZE


class NobitexDataFetcher:
    """
    Fetches and processes historical market data from Nobitex API.
    Creates features for LSTM model input.
    """
    
    def __init__(self, api: NobitexAPI):
        self.api = api
        self.window_size = WINDOW_SIZE
    
    def fetch_historical_data(self, pair: str, days: int = 7) -> pd.DataFrame:
        """
        Fetch historical market data using the UDF history endpoint.
        
        Args:
            pair: Trading pair (e.g., 'WIRT')
            days: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching data for {pair} (last {days} days)...")
        
        try:
            udf_data = self.api.get_historical_data(pair, resolution="60", days=days)
            
            if udf_data.get('s') != 'ok':
                raise ValueError(f"API returned error status: {udf_data.get('s')}")
            
            timestamps = udf_data.get('t', [])
            opens = udf_data.get('o', [])
            highs = udf_data.get('h', [])
            lows = udf_data.get('l', [])
            closes = udf_data.get('c', [])
            volumes = udf_data.get('v', [])
            
            if not timestamps or len(timestamps) == 0:
                raise ValueError(f"No historical data available for {pair}")
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps, unit='s'),
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })
            
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            print(f"Fetched {len(df)} data points for {pair}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            print(f"Price range: {df['close'].min():,.0f} - {df['close'].max():,.0f} IRT")
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {pair}: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, object]:
        """
        Prepare features for LSTM model input.
        
        Features: [open, high, low, close, volume, position]
        Normalized to match the training environment's observation space.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Tuple of (normalized feature array, scaler object)
        """
        print("Preparing features to match training environment...")
        
        features = ['open', 'high', 'low', 'close', 'volume']
        feature_data = df[features].values
        
        normalized_features = np.zeros((len(feature_data), 6), dtype=np.float32)
        
        for i in range(len(feature_data)):
            current_close = feature_data[i, 3]
            
            if current_close > 1e-8:
                normalized_features[i, :4] = feature_data[i, :4] / current_close
                
                max_volume = np.max(feature_data[:, 4])
                if max_volume > 1e-8:
                    normalized_features[i, 4] = feature_data[i, 4] / max_volume
                else:
                    normalized_features[i, 4] = 0.0
            else:
                normalized_features[i, :5] = 0.0
            
            normalized_features[i, 5] = 0.0  # Position indicator
        
        class SimpleScaler:
            def __init__(self, mean, std):
                self.mean_ = mean
                self.scale_ = std
        
        mean = np.mean(normalized_features, axis=0)
        std = np.std(normalized_features, axis=0)
        scaler = SimpleScaler(mean, std)
        
        print(f"Features prepared: {features + ['position']}")
        print(f"Feature shape: {normalized_features.shape}")
        
        return normalized_features, scaler
