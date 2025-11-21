"""
Nobitex API Client Module
=========================
Handles all API communication with Nobitex exchange.
"""

import time
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class NobitexConfig:
    """Nobitex API configuration"""
    BASE_URL: str = "https://apiv2.nobitex.ir"
    RATE_LIMIT_DELAY: float = 0.6  # 100 req/min = 0.6s between requests
    REQUEST_TIMEOUT: int = 10
    MAX_RETRIES: int = 3


class NobitexAPI:
    """
    Nobitex exchange API client.
    Handles rate limiting, retries, and data normalization.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.config = NobitexConfig()
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NobitexDQNEvaluator/1.0',
            'Content-Type': 'application/json'
        })
        if api_key:
            self.session.headers.update({'Authorization': f'Token {api_key}'})
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limiting and retry logic."""
        url = f"{self.config.BASE_URL}{endpoint}"
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                time.sleep(self.config.RATE_LIMIT_DELAY)
                
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=self.config.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                return response.json()
                    
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise Exception(f"All retry attempts failed: {e}")
        
        raise Exception("Maximum retries exceeded")
    
    def get_historical_data(self, pair: str, resolution: str = "60", days: int = 7) -> Dict:
        """
        Get historical OHLCV data using UDF history endpoint.
        
        Args:
            pair: Trading pair (e.g., 'WIRT')
            resolution: Time resolution in minutes (60 = 1 hour)
            days: Number of days of historical data
            
        Returns:
            UDF format data with t, o, h, l, c, v arrays
        """
        end_time = int(time.time())
        start_time = end_time - (days * 24 * 60 * 60)
        
        params = {
            'symbol': pair,
            'resolution': resolution,
            'from': start_time,
            'to': end_time
        }
        
        return self._make_request("/market/udf/history", params)
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available trading pairs."""
        return ['WIRT', 'ETHIRT', 'LTCIRT', 'XRPIRT', 'ADAIRT', 'DOGEIRT', 'DOTIRT', 'UNIIRT']
