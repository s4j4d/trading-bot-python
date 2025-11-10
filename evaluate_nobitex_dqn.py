#!/usr/bin/env python3
"""
Nobitex DQN Model Evaluation Script
==================================

Complete backtesting evaluation for trained DQN (LSTM) model on Nobitex exchange.
Fetches real market data via Nobitex API and evaluates trading performance.

Author: Expert Python RL Trading Engineer
Exchange: Nobitex (Iran's top crypto exchange)
Model: DQN with LSTM for temporal pattern recognition
Pairs: WIRT, ETHIRT, etc. (crypto vs Iranian Rial/Toman)

Usage:
    python evaluate_nobitex_dqn.py --pair WIRT --days 7
    python evaluate_nobitex_dqn.py --pair ETHIRT --days 30 --api-key YOUR_KEY
"""

import argparse
import time
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
from config.constants import (
    WINDOW_SIZE, INITIAL_BALANCE, FEE_RATE, SLIPPAGE_FACTOR
)
from model.q_network import build_q_network

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

@dataclass
class NobitexConfig:
    """Nobitex API configuration and trading parameters"""
    BASE_URL: str = "https://apiv2.nobitex.ir"
    RATE_LIMIT_DELAY: float = 0.6  # 100 req/min = 0.6s between requests
    REQUEST_TIMEOUT: int = 10
    MAX_RETRIES: int = 3
    
    # Trading parameters from constants.py
    # INITIAL_FIAT_BALANCE: float = INITIAL_FIAT_BALANCE
    INITIAL_BALANCE: float = INITIAL_BALANCE
    # INITIAL_CRYPTO_BALANCE: float = 0.0
    FEE_RATE: float = FEE_RATE
    SLIPPAGE_FACTOR: float = SLIPPAGE_FACTOR
    WINDOW_SIZE: int = WINDOW_SIZE
    
    # Performance thresholds
    PROFITABLE_THRESHOLD: float = 0.20  # Strategy must beat HODL by 20%
    MAX_DRAWDOWN_THRESHOLD: float = 0.30  # Max acceptable drawdown 30%

# ============================================================================
# NOBITEX API CLIENT
# ============================================================================

class NobitexAPI:
    """
    Nobitex exchange API client using only the /market/udf/history endpoint.
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
        """
        Make API request with rate limiting and retry logic.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response data
            
        Raises:
            Exception: If all retries fail
        """
        url = f"{self.config.BASE_URL}{endpoint}"
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                # Rate limiting
                time.sleep(self.config.RATE_LIMIT_DELAY)
                
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=self.config.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                
                # UDF endpoint returns data directly, not wrapped in status
                data = response.json()
                return data
                    
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
        # Calculate timestamps
        end_time = int(time.time())
        start_time = end_time - (days * 24 * 60 * 60)  # days ago
        
        params = {
            'symbol': pair,
            'resolution': resolution,
            'from': start_time,
            'to': end_time
        }
        
        return self._make_request("/market/udf/history", params)
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available trading pairs (fallback list since we only use UDF endpoint)"""
        # Since we can't query available pairs, return common Nobitex pairs
        return ['WIRT', 'ETHIRT', 'LTCIRT', 'XRPIRT', 'ADAIRT', 'DOGEIRT', 'DOTIRT', 'UNIIRT']

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Array of closing prices
        period: RSI calculation period
        
    Returns:
        RSI values (0-100)
    """
    if len(prices) < period + 1:
        return np.full(len(prices), 50.0)  # Neutral RSI for insufficient data
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    rsi = np.zeros(len(prices))
    rsi[:period] = 50.0  # Neutral for initial values
    
    # Calculate RSI for remaining periods
    for i in range(period, len(prices)):
        if i == period:
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Array of closing prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period
        
    Returns:
        MACD line values
    """
    if len(prices) < slow:
        return np.zeros(len(prices))
    
    # Calculate EMAs
    ema_fast = pd.Series(prices).ewm(span=fast).mean().values
    ema_slow = pd.Series(prices).ewm(span=slow).mean().values
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line
    signal_line = pd.Series(macd_line).ewm(span=signal).mean().values
    
    # Return MACD line (could also return histogram: macd_line - signal_line)
    return macd_line

# ============================================================================
# DATA FETCHER & PROCESSOR
# ============================================================================

class NobitexDataFetcher:
    """
    Fetches and processes historical market data from Nobitex API.
    Creates features for LSTM model input.
    """
    
    def __init__(self, api: NobitexAPI):
        self.api = api
        self.config = NobitexConfig()
    
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
            # Get historical data from UDF endpoint
            udf_data = self.api.get_historical_data(pair, resolution="60", days=days)
            
            # Check if data is valid
            if udf_data.get('s') != 'ok':
                raise ValueError(f"API returned error status: {udf_data.get('s')}")
            
            # Extract OHLCV arrays from UDF format
            timestamps = udf_data.get('t', [])
            opens = udf_data.get('o', [])
            highs = udf_data.get('h', [])
            lows = udf_data.get('l', [])
            closes = udf_data.get('c', [])
            volumes = udf_data.get('v', [])
            
            if not timestamps or len(timestamps) == 0:
                raise ValueError(f"No historical data available for {pair}")
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps, unit='s'),
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Sort by timestamp (oldest first)
            df.sort_index(inplace=True)
            
            print(f"Fetched {len(df)} data points for {pair}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            print(f"Price range: {df['close'].min():,.0f} - {df['close'].max():,.0f} IRT")
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {pair}: {e}")
            raise
    

    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for LSTM model input.
        
        Features: [close, volume, RSI(14), MACD]
        Normalized using StandardScaler for stable training.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Normalized feature array
        """
        print("Calculating technical indicators...")
        
        # Calculate technical indicators
        df['rsi'] = calculate_rsi(df['close'].values)
        df['macd'] = calculate_macd(df['close'].values)
        
        # Select features for model - using only close prices to match model architecture
        # The LSTM model was trained on normalized close prices only
        features = ['close']  # Only close prices for now
        feature_data = df[features].values
        
        # Normalize features using z-score normalization (mean=0, std=1)
        # This replaces sklearn.StandardScaler to avoid external dependency
        mean = np.mean(feature_data, axis=0)
        std = np.std(feature_data, axis=0)
        # Avoid division by zero for constant features
        std = np.where(std == 0, 1, std)
        normalized_features = (feature_data - mean) / std
        
        # Create a simple scaler object for consistency
        class SimpleScaler:
            def __init__(self, mean, std):
                self.mean_ = mean
                self.scale_ = std
        
        scaler = SimpleScaler(mean, std)
        
        print(f"Features prepared: {features}")
        print(f"Feature shape: {normalized_features.shape}")
        print(f"Note: Using only close prices to match LSTM model architecture")
        
        return normalized_features, scaler

# ============================================================================
# BACKTESTER
# ============================================================================

@dataclass
class BacktestResult:
    """Container for backtest results and performance metrics"""
    total_pnl_irt: float
    total_pnl_pct: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    profit_factor: float
    win_rate: float
    total_trades: int
    equity_curve: List[float]
    hodl_curve: List[float]
    drawdown_curve: List[float]
    trade_log: List[Dict]

class NobitexBacktester:
    """
    Backtesting engine for DQN trading strategy on Nobitex data.
    Implements realistic trading simulation with fees and slippage.
    """
    
    def __init__(self, model, config: NobitexConfig):
        self.model = model
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset backtester state for new run"""
        self.balance = self.config.INITIAL_BALANCE
        self.position = 0.0  # Crypto holdings
        self.equity_curve = []
        self.trade_log = []
        self.current_step = 0
    
    def calculate_portfolio_value(self, price: float) -> float:
        """Calculate total portfolio value in IRT"""
        return self.balance + (self.position * price)
    
    def execute_trade(self, action: int, price: float, timestamp: str) -> bool:
        """
        Execute trading action with realistic fees and slippage.
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell
            price: Current market price
            timestamp: Current timestamp
            
        Returns:
            True if trade was executed
        """
        if action == 0:  # Hold
            return False
        
        # Apply slippage
        if action == 1:  # Buy
            execution_price = price * (1 + self.config.SLIPPAGE_FACTOR)
        else:  # Sell
            execution_price = price * (1 - self.config.SLIPPAGE_FACTOR)
        
        trade_executed = False
        
        if action == 1 and self.position == 0:  # Buy (enter long position)
            # Use 100% of balance for spot trading
            gross_amount = self.balance / execution_price
            fee = gross_amount * self.config.FEE_RATE
            net_amount = gross_amount - fee
            
            if net_amount > 0:
                self.position = net_amount
                self.balance = 0.0
                trade_executed = True
                
                self.trade_log.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': execution_price,
                    'amount': net_amount,
                    'fee': fee * execution_price,
                    'portfolio_value': self.calculate_portfolio_value(price)
                })
        
        elif action == 2 and self.position > 0:  # Sell (exit long position)
            gross_proceeds = self.position * execution_price
            fee = gross_proceeds * self.config.FEE_RATE
            net_proceeds = gross_proceeds - fee
            
            self.balance = net_proceeds
            self.position = 0.0
            trade_executed = True
            
            self.trade_log.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'price': execution_price,
                'amount': self.position,
                'fee': fee,
                'portfolio_value': self.calculate_portfolio_value(price)
            })
        
        return trade_executed
    
    def run_backtest(self, df: pd.DataFrame, features: np.ndarray) -> BacktestResult:
        """
        Run complete backtest on historical data.
        
        Args:
            df: OHLCV DataFrame
            features: Normalized feature array
            
        Returns:
            BacktestResult with performance metrics
        """
        print("Running backtest...")
        self.reset()
        
        # Skip initial steps for LSTM warmup (reduce warmup for short datasets)
        warmup_steps = min(50, len(df) // 4)  # Use 50 or 1/4 of data, whichever is smaller
        start_idx = max(warmup_steps, self.config.WINDOW_SIZE)
        
        print(f"Skipping first {start_idx} steps for LSTM warmup")
        print(f"Total data points: {len(df)}")
        print(f"Will process steps {start_idx} to {len(df)-1} ({len(df) - start_idx} steps)")
        print(f"WINDOW_SIZE: {self.config.WINDOW_SIZE}")
        
        if len(df) - start_idx < 10:
            print(f"WARNING: Very few steps to process ({len(df) - start_idx}). Consider using more days of data.")
        
        # Initialize with starting portfolio value
        initial_price = df.iloc[start_idx]['close']
        initial_portfolio_value = self.config.INITIAL_BALANCE
        
        # Track equity curve and HODL benchmark
        equity_curve = [initial_portfolio_value]
        hodl_curve = [initial_portfolio_value]
        
        # HODL benchmark: buy and hold from start
        hodl_crypto_amount = initial_portfolio_value / initial_price
        
        print(f"Initial setup: Starting with {self.config.INITIAL_BALANCE:,.0f} IRT cash")
        print(f"Initial price: {initial_price:.0f} IRT")
        
        # Force initial buy to give model crypto to trade with
        print("Forcing initial crypto purchase...")
        initial_timestamp = df.index[start_idx].strftime('%Y-%m-%d %H:%M:%S')
        self.execute_trade(1, initial_price, initial_timestamp)  # Force buy action
        print(f"Initial position: {self.position:.6f} crypto, Balance: {self.balance:.0f} IRT")
        
        # Track action counts for debugging
        action_counts = {0: 0, 1: 0, 2: 0}  # Hold, Buy, Sell
        
        # Run backtest
        print(f"Starting backtest loop from {start_idx} to {len(df)-1}")
        for i in range(start_idx, len(df)):
            current_price = df.iloc[i]['close']
            timestamp = df.index[i].strftime('%Y-%m-%d %H:%M:%S')
            
            # Debug: Print every 10th step
            if i % 10 == 0:
                print(f"Processing step {i}/{len(df)-1}, Price: {current_price:.0f}")
            
            # Prepare state for model (last WINDOW_SIZE close prices)
            if i >= self.config.WINDOW_SIZE:
                # Extract last WINDOW_SIZE normalized close prices
                state = features[i-self.config.WINDOW_SIZE:i, 0]  # Take only close prices (first column)
                state = np.expand_dims(state, axis=0)  # Add batch dimension: (1, WINDOW_SIZE)
                
                # Debug: Print state info for first few steps
                if i <= start_idx + 2:
                    print(f"Step {i}: State shape={state.shape}, State sample={state[0][:5]}...")
                
                # Get model prediction
                q_values = self.model.predict(state, verbose=0)
                action = np.argmax(q_values[0])
                action_counts[action] += 1
                
                # Debug: Print first few predictions to see what model is doing
                if i <= start_idx + 5:  # Print first 5 predictions
                    print(f"Step {i}: Q-values={q_values[0]}, Action={action} ({'Hold' if action==0 else 'Buy' if action==1 else 'Sell'}), Price={current_price:.0f}")
                
                # Execute trade
                trade_executed = self.execute_trade(action, current_price, timestamp)
                
                # Debug: Print when trades are executed
                if trade_executed:
                    print(f"TRADE EXECUTED at step {i}: Action={action}, Price={current_price:.0f}")
            else:
                # Debug: Show why we're skipping
                if i < start_idx + 3:
                    print(f"Step {i}: Skipping (i={i} < WINDOW_SIZE={self.config.WINDOW_SIZE})")
            
            # Update curves
            portfolio_value = self.calculate_portfolio_value(current_price)
            equity_curve.append(portfolio_value)
            
            hodl_value = hodl_crypto_amount * current_price
            hodl_curve.append(hodl_value)
        
        # Print action summary
        total_predictions = sum(action_counts.values())
        if total_predictions > 0:
            print(f"\nAction Summary:")
            print(f"Hold: {action_counts[0]} ({action_counts[0]/total_predictions*100:.1f}%)")
            print(f"Buy:  {action_counts[1]} ({action_counts[1]/total_predictions*100:.1f}%)")
            print(f"Sell: {action_counts[2]} ({action_counts[2]/total_predictions*100:.1f}%)")
            print(f"Total predictions: {total_predictions}")
        
        # Calculate performance metrics
        result = self._calculate_metrics(
            equity_curve, hodl_curve, df.iloc[start_idx:], initial_portfolio_value
        )
        
        print(f"Backtest completed: {len(self.trade_log)} trades executed")
        return result
    
    def _calculate_metrics(self, equity_curve: List[float], hodl_curve: List[float], 
                          df: pd.DataFrame, initial_value: float) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        
        equity_array = np.array(equity_curve)
        hodl_array = np.array(hodl_curve)
        
        # Basic P&L
        final_value = equity_array[-1]
        total_pnl_irt = final_value - initial_value
        total_pnl_pct = (final_value / initial_value - 1) * 100
        
        # Time-based metrics
        days = len(df) / 24  # Assuming hourly data
        years = days / 365.25
        
        # CAGR (Compound Annual Growth Rate)
        if years > 0:
            cagr = (final_value / initial_value) ** (1/years) - 1
        else:
            cagr = 0.0
        
        # Returns for ratio calculations
        returns = np.diff(equity_array) / equity_array[:-1]
        returns = returns[~np.isnan(returns)]  # Remove NaN values
        
        # Sharpe Ratio (annualized, risk-free rate = 0)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365.25 * 24)  # Hourly to annual
        else:
            sharpe_ratio = 0.0
        
        # Sortino Ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 1:
            downside_std = np.std(negative_returns)
            if downside_std > 0:
                sortino_ratio = np.mean(returns) / downside_std * np.sqrt(365.25 * 24)
            else:
                sortino_ratio = sharpe_ratio
        else:
            sortino_ratio = sharpe_ratio
        
        # Maximum Drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_drawdown = abs(np.min(drawdown)) * 100  # Convert to percentage
        drawdown_curve = drawdown.tolist()
        
        # Calmar Ratio
        if max_drawdown > 0:
            calmar_ratio = (cagr * 100) / max_drawdown
        else:
            calmar_ratio = 0.0
        
        # Trade-based metrics
        if self.trade_log:
            # Profit Factor
            winning_trades = [t for t in self.trade_log if t['action'] == 'SELL']
            if len(winning_trades) >= 2:
                # Calculate P&L for each trade pair
                buy_trades = [t for t in self.trade_log if t['action'] == 'BUY']
                profits = []
                losses = []
                
                for i in range(min(len(buy_trades), len(winning_trades))):
                    buy_price = buy_trades[i]['price']
                    sell_price = winning_trades[i]['price']
                    pnl = (sell_price - buy_price) / buy_price
                    
                    if pnl > 0:
                        profits.append(pnl)
                    else:
                        losses.append(abs(pnl))
                
                total_profit = sum(profits) if profits else 0
                total_loss = sum(losses) if losses else 1e-10
                profit_factor = total_profit / total_loss if total_loss > 0 else 0
                win_rate = len(profits) / (len(profits) + len(losses)) * 100 if (profits or losses) else 0
            else:
                profit_factor = 0.0
                win_rate = 0.0
            
            total_trades = len(self.trade_log)
        else:
            profit_factor = 0.0
            win_rate = 0.0
            total_trades = 0
        
        return BacktestResult(
            total_pnl_irt=total_pnl_irt,
            total_pnl_pct=total_pnl_pct,
            cagr=cagr * 100,  # Convert to percentage
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            profit_factor=profit_factor,
            win_rate=win_rate,
            total_trades=total_trades,
            equity_curve=equity_curve,
            hodl_curve=hodl_curve,
            drawdown_curve=drawdown_curve,
            trade_log=self.trade_log
        )

# ============================================================================
# VISUALIZATION & REPORTING
# ============================================================================

def plot_results(result: BacktestResult, pair: str, save_path: str = None):
    """
    Create comprehensive performance visualization.
    
    Args:
        result: Backtest results
        pair: Trading pair name
        save_path: Optional path to save plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'DQN Trading Performance - {pair}', fontsize=16, fontweight='bold')
    
    # 1. Equity Curve vs HODL
    ax1.plot(result.equity_curve, label='DQN Strategy', linewidth=2, color='blue')
    ax1.plot(result.hodl_curve, label='HODL Benchmark', linewidth=2, color='orange')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Time Steps (Hours)')
    ax1.set_ylabel('Portfolio Value (IRT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')
    
    # 2. Drawdown Curve
    ax2.fill_between(range(len(result.drawdown_curve)), result.drawdown_curve, 0, 
                     alpha=0.7, color='red', label='Drawdown')
    ax2.set_title('Drawdown Curve')
    ax2.set_xlabel('Time Steps (Hours)')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance Metrics Bar Chart
    metrics = ['CAGR (%)', 'Sharpe', 'Sortino', 'Max DD (%)', 'Win Rate (%)']
    values = [result.cagr, result.sharpe_ratio, result.sortino_ratio, 
              -result.max_drawdown, result.win_rate]
    colors = ['green' if v >= 0 else 'red' for v in values]
    
    bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
    ax3.set_title('Performance Metrics')
    ax3.set_ylabel('Value')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.01 * max(abs(v) for v in values)),
                f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # 4. Trade Distribution (if trades exist)
    if result.trade_log:
        buy_trades = [t for t in result.trade_log if t['action'] == 'BUY']
        sell_trades = [t for t in result.trade_log if t['action'] == 'SELL']
        
        ax4.scatter(range(len(buy_trades)), [t['price'] for t in buy_trades], 
                   color='green', marker='^', s=50, label='Buy', alpha=0.7)
        ax4.scatter(range(len(sell_trades)), [t['price'] for t in sell_trades], 
                   color='red', marker='v', s=50, label='Sell', alpha=0.7)
        ax4.set_title('Trade Execution Points')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('Price (IRT)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No Trades Executed', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Trade Execution Points')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def print_performance_table(result: BacktestResult, pair: str):
    """
    Print comprehensive performance metrics table.
    
    Args:
        result: Backtest results
        pair: Trading pair name
    """
    print(f"\n{'='*60}")
    print(f"PERFORMANCE REPORT - {pair}")
    print(f"{'='*60}")
    
    # Calculate HODL performance for comparison
    hodl_pnl_pct = (result.hodl_curve[-1] / result.hodl_curve[0] - 1) * 100
    hodl_cagr = ((result.hodl_curve[-1] / result.hodl_curve[0]) ** (365.25 / (len(result.hodl_curve) / 24)) - 1) * 100
    
    print(f"{'Metric':<25} {'Strategy':<15} {'HODL':<15} {'Difference':<15}")
    print(f"{'-'*70}")
    print(f"{'Total PnL (IRT)':<25} {result.total_pnl_irt:>14,.0f} {result.hodl_curve[-1] - result.hodl_curve[0]:>14,.0f} {result.total_pnl_irt - (result.hodl_curve[-1] - result.hodl_curve[0]):>14,.0f}")
    print(f"{'Total PnL (%)':<25} {result.total_pnl_pct:>14.2f} {hodl_pnl_pct:>14.2f} {result.total_pnl_pct - hodl_pnl_pct:>14.2f}")
    print(f"{'CAGR (%)':<25} {result.cagr:>14.2f} {hodl_cagr:>14.2f} {result.cagr - hodl_cagr:>14.2f}")
    print(f"{'Sharpe Ratio':<25} {result.sharpe_ratio:>14.2f} {'N/A':<15} {'N/A':<15}")
    print(f"{'Sortino Ratio':<25} {result.sortino_ratio:>14.2f} {'N/A':<15} {'N/A':<15}")
    print(f"{'Max Drawdown (%)':<25} {result.max_drawdown:>14.2f} {'N/A':<15} {'N/A':<15}")
    print(f"{'Calmar Ratio':<25} {result.calmar_ratio:>14.2f} {'N/A':<15} {'N/A':<15}")
    print(f"{'Profit Factor':<25} {result.profit_factor:>14.2f} {'N/A':<15} {'N/A':<15}")
    print(f"{'Win Rate (%)':<25} {result.win_rate:>14.2f} {'N/A':<15} {'N/A':<15}")
    print(f"{'Total Trades':<25} {result.total_trades:>14d} {'1':<15} {'N/A':<15}")
    
    print(f"\n{'PROFITABILITY ANALYSIS'}")
    print(f"{'-'*30}")
    
    # Determine if strategy is profitable
    outperformance = result.cagr - hodl_cagr
    is_profitable = (outperformance > 20.0 and result.max_drawdown < 30.0)
    
    print(f"Strategy vs HODL: {outperformance:+.2f}% CAGR difference")
    print(f"Max Drawdown: {result.max_drawdown:.2f}% ({'ACCEPTABLE' if result.max_drawdown < 30 else 'HIGH RISK'})")
    print(f"Profitability Threshold: >20% outperformance + <30% max drawdown")
    
    verdict = "YES" if is_profitable else "NO"
    print(f"\nStrategy profitable vs HODL: {verdict}")
    
    if result.trade_log:
        print(f"\nTRADE SUMMARY:")
        print(f"First Trade: {result.trade_log[0]['timestamp']}")
        print(f"Last Trade: {result.trade_log[-1]['timestamp']}")
        print(f"Average Trade Size: {np.mean([t.get('amount', 0) for t in result.trade_log]):.4f} crypto")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def load_trained_model(model_path: str = "final_parallel_trading_model.keras") -> tf.keras.Model:
    """
    Load the trained DQN model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded TensorFlow model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    try:
        print(f"Loading trained model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return model
        
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Please ensure the model file exists or train a new model first.")
        raise
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("The model file may be corrupted or incompatible.")
        raise

def main():
    """Main execution function with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="Evaluate DQN trading model on Nobitex exchange",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_nobitex_dqn.py --pair WIRT --days 7
  python evaluate_nobitex_dqn.py --pair ETHIRT --days 30 --api-key YOUR_KEY
  python evaluate_nobitex_dqn.py --pair WIRT --days 14 --model custom_model.keras
        """
    )
    
    parser.add_argument('--pair', type=str, default='WIRT',
                       help='Trading pair (default: WIRT)')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days for backtesting (default: 7)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Nobitex API key (optional, for higher rate limits)')
    parser.add_argument('--model', type=str, default='final_parallel_trading_model.keras',
                       help='Path to trained model file')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Path to save performance plot')
    
    args = parser.parse_args()
    
    print("="*60)
    print("NOBITEX DQN MODEL EVALUATION")
    print("="*60)
    print(f"Pair: {args.pair}")
    print(f"Evaluation Period: {args.days} days")
    print(f"Model: {args.model}")
    print(f"Initial Balance: {INITIAL_BALANCE:,.0f} IRT")
    print(f"Fee Rate: {FEE_RATE*100:.2f}%")
    print(f"Slippage: {SLIPPAGE_FACTOR*100:.3f}%")
    print("="*60)
    
    try:
        # Initialize API client
        api = NobitexAPI(api_key=args.api_key)
        
        # Check if pair is available
        available_pairs = api.get_available_pairs()
        if args.pair not in available_pairs:
            print(f"Warning: {args.pair} not found in available pairs.")
            print(f"Available pairs: {', '.join(available_pairs[:10])}...")
            print("Continuing with specified pair...")
        
        # Load trained model
        model = load_trained_model(args.model)
        
        # Fetch and prepare data
        data_fetcher = NobitexDataFetcher(api)
        df = data_fetcher.fetch_historical_data(args.pair, args.days)
        features, scaler = data_fetcher.prepare_features(df)
        
        # Initialize backtester
        config = NobitexConfig()
        backtester = NobitexBacktester(model, config)
        
        # Run backtest
        result = backtester.run_backtest(df, features)
        
        # Display results
        print_performance_table(result, args.pair)
        
        # Create visualization
        plot_results(result, args.pair, args.save_plot)
        
        # Final verdict
        hodl_cagr = ((result.hodl_curve[-1] / result.hodl_curve[0]) ** (365.25 / args.days) - 1) * 100
        outperformance = result.cagr - hodl_cagr
        is_profitable = (outperformance > config.PROFITABLE_THRESHOLD * 100 and 
                        result.max_drawdown < config.MAX_DRAWDOWN_THRESHOLD * 100)
        
        print("\n" + "="*60)
        print("FINAL VERDICT")
        print("="*60)
        print(f"Strategy CAGR: {result.cagr:.2f}%")
        print(f"HODL CAGR: {hodl_cagr:.2f}%")
        print(f"Outperformance: {outperformance:+.2f}%")
        print(f"Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"Required: >{config.PROFITABLE_THRESHOLD*100:.0f}% outperformance, <{config.MAX_DRAWDOWN_THRESHOLD*100:.0f}% drawdown")
        print(f"\nStrategy profitable vs HODL: {'YES' if is_profitable else 'NO'}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())