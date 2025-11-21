"""
Visualization and Reporting Module
==================================
Creates charts and performance reports.
"""

import numpy as np
import matplotlib.pyplot as plt
from evaluation.backtester import BacktestResult


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
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.01 * max(abs(v) for v in values)),
                f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # 4. Trade Distribution
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
