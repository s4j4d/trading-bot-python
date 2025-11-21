#!/usr/bin/env python3
"""
Nobitex DQN Model Evaluation Script
==================================

Main evaluation script for trained DQN model on Nobitex exchange.
This is the simplified entry point - all complex logic is in separate modules.

Usage:
    python evaluate_nobitex_dqn.py --pair WIRT --days 7
    python evaluate_nobitex_dqn.py --pair ETHIRT --days 30 --api-key YOUR_KEY
"""

import sys
import os

import argparse
import warnings
warnings.filterwarnings('ignore')
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.constants import (INITIAL_BALANCE, FEE_RATE, SLIPPAGE_FACTOR)
from evaluation.nobitex_api import NobitexAPI
from evaluation.data_processor import NobitexDataFetcher
from evaluation.backtester import NobitexBacktester
from evaluation.visualization import plot_results, print_performance_table
from evaluation.model_loader import load_trained_model


def main():
    """Main execution function - simplified and clean"""
    
    # Parse command line arguments
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
    parser.add_argument('--initial-fiat', type=float, default=None,
                       help=f'Initial fiat balance in IRT (default: {INITIAL_BALANCE:,.0f})')
    parser.add_argument('--initial-crypto', type=float, default=None,
                       help='Initial crypto holdings (default: 0.0)')
    
    args = parser.parse_args()
    
    # Print configuration
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
        # Step 1: Initialize API client
        print("\n[1/5] Initializing API client...")
        api = NobitexAPI(api_key=args.api_key)
        
        # Step 2: Load trained model
        print("\n[2/5] Loading trained model...")
        model = load_trained_model(args.model)
        
        # Step 3: Fetch and prepare data
        print("\n[3/5] Fetching market data...")
        data_fetcher = NobitexDataFetcher(api)
        df = data_fetcher.fetch_historical_data(args.pair, args.days)
        features, scaler = data_fetcher.prepare_features(df)
        
        # Step 4: Run backtest
        print("\n[4/5] Running backtest...")
        initial_fiat = args.initial_fiat if args.initial_fiat is not None else INITIAL_BALANCE
        initial_crypto = args.initial_crypto if args.initial_crypto is not None else 0.0
        
        backtester = NobitexBacktester(model, initial_fiat, initial_crypto)
        result = backtester.run_backtest(df, features)
        
        # Step 5: Display results
        print("\n[5/5] Generating results...")
        print_performance_table(result, args.pair)
        plot_results(result, args.pair, args.save_plot)
        
        # Final verdict
        hodl_cagr = ((result.hodl_curve[-1] / result.hodl_curve[0]) ** (365.25 / args.days) - 1) * 100
        outperformance = result.cagr - hodl_cagr
        is_profitable = (outperformance > 20.0 and result.max_drawdown < 30.0)
        
        print("\n" + "="*60)
        print("FINAL VERDICT")
        print("="*60)
        print(f"Strategy CAGR: {result.cagr:.2f}%")
        print(f"HODL CAGR: {hodl_cagr:.2f}%")
        print(f"Outperformance: {outperformance:+.2f}%")
        print(f"Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"Required: >20% outperformance, <30% drawdown")
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
