"""
Backtesting Engine Module
=========================
Simulates trading with realistic fees and slippage.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass
from config.constants import INITIAL_BALANCE, FEE_RATE, SLIPPAGE_FACTOR, WINDOW_SIZE


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
    Backtesting engine for DQN trading strategy.
    Implements realistic trading simulation with fees and slippage.
    """
    
    def __init__(self, model, initial_fiat: float = INITIAL_BALANCE, initial_crypto: float = 0.0):
        self.model = model
        self.initial_fiat = initial_fiat
        self.initial_crypto = initial_crypto
        self.fee_rate = FEE_RATE
        self.slippage = SLIPPAGE_FACTOR
        self.window_size = WINDOW_SIZE
        self.reset()
    
    def reset(self):
        """Reset backtester state for new run"""
        self.balance = self.initial_fiat
        self.position = self.initial_crypto
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
        if action == 0:
            return False
        
        if action == 1:
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)
        
        trade_executed = False
        
        if action == 1 and self.position == 0:
            gross_amount = self.balance / execution_price
            fee = gross_amount * self.fee_rate
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
        
        elif action == 2 and self.position > 0:
            gross_proceeds = self.position * execution_price
            fee = gross_proceeds * self.fee_rate
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
        print(features)
        self.reset()
        
        warmup_steps = min(50, len(df) // 4)
        start_idx = max(warmup_steps, self.window_size)
        
        print(f"Skipping first {start_idx} steps for LSTM warmup")
        print(f"Will process steps {start_idx} to {len(df)-1} ({len(df) - start_idx} steps)")
        
        initial_price = df.iloc[start_idx]['close']
        initial_portfolio_value = self.initial_fiat + (self.initial_crypto * initial_price)
        
        equity_curve = [initial_portfolio_value]
        hodl_curve = [initial_portfolio_value]
        hodl_crypto_amount = initial_portfolio_value / initial_price
        
        print(f"Initial Portfolio Value: {initial_portfolio_value:,.0f} IRT")
        
        if self.initial_crypto == 0.0 and self.initial_fiat > 0:
            print("\nForcing initial crypto purchase...")
            initial_timestamp = df.index[start_idx].strftime('%Y-%m-%d %H:%M:%S')
            self.execute_trade(1, initial_price, initial_timestamp)
        
        action_counts = {0: 0, 1: 0, 2: 0}
        
        for i in range(start_idx, len(df)):
            current_price = df.iloc[i]['close']
            timestamp = df.index[i].strftime('%Y-%m-%d %H:%M:%S')
            
            if i >= self.window_size:
                state = features[i-self.window_size:i, :].copy()
                position_indicator = 1.0 if self.position > 1e-8 else 0.0
                state[:, 5] = position_indicator
                state = np.expand_dims(state, axis=0)
                
                q_values = self.model.predict(state, verbose=0)
                print(f"#q_values for this window:{q_values}")
                action = np.argmax(q_values[0])
                action_counts[action] += 1
                
                self.execute_trade(action, current_price, timestamp)
            
            portfolio_value = self.calculate_portfolio_value(current_price)
            equity_curve.append(portfolio_value)
            
            hodl_value = hodl_crypto_amount * current_price
            hodl_curve.append(hodl_value)
        
        total_predictions = sum(action_counts.values())
        if total_predictions > 0:
            print(f"\nAction Summary:")
            print(f"Hold: {action_counts[0]} ({action_counts[0]/total_predictions*100:.1f}%)")
            print(f"Buy:  {action_counts[1]} ({action_counts[1]/total_predictions*100:.1f}%)")
            print(f"Sell: {action_counts[2]} ({action_counts[2]/total_predictions*100:.1f}%)")
        
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
        
        final_value = equity_array[-1]
        total_pnl_irt = final_value - initial_value
        total_pnl_pct = (final_value / initial_value - 1) * 100
        
        days = len(df) / 24
        years = days / 365.25
        
        if years > 0:
            cagr = (final_value / initial_value) ** (1/years) - 1
        else:
            cagr = 0.0
        
        returns = np.diff(equity_array) / equity_array[:-1]
        returns = returns[~np.isnan(returns)]
        
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365.25 * 24)
        else:
            sharpe_ratio = 0.0
        
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 1:
            downside_std = np.std(negative_returns)
            if downside_std > 0:
                sortino_ratio = np.mean(returns) / downside_std * np.sqrt(365.25 * 24)
            else:
                sortino_ratio = sharpe_ratio
        else:
            sortino_ratio = sharpe_ratio
        
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_drawdown = abs(np.min(drawdown)) * 100
        drawdown_curve = drawdown.tolist()
        
        if max_drawdown > 0:
            calmar_ratio = (cagr * 100) / max_drawdown
        else:
            calmar_ratio = 0.0
        
        if self.trade_log:
            winning_trades = [t for t in self.trade_log if t['action'] == 'SELL']
            if len(winning_trades) >= 2:
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
            cagr=cagr * 100,
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
