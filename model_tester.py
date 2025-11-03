"""
Model Testing Module for Cryptocurrency Trading Bot

This module loads a trained DQN model and tests its performance against actual market data.
It provides comprehensive evaluation metrics and visualizations to assess the model's
trading performance compared to buy-and-hold strategies.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from datetime import datetime
import warnings

# Import project modules
from data import load_data_from_json
from environment import CryptoTradingEnv
from config.constants import (
    WINDOW_SIZE,
    INITIAL_BALANCE,
    MAX_STEPS_PER_EPISODE,
    JSON_FILE_PATH
)


class ModelTester:
    """
    A comprehensive testing class for evaluating trained DQN trading models.
    
    This class provides functionality to:
    - Load trained models
    - Run trading simulations on test data
    - Calculate performance metrics
    - Generate trading reports and visualizations
    """
    
    def __init__(self, model_path="final_parallel_trading_model.keras"):
        """
        Initialize the model tester.
        
        Args:
            model_path (str): Path to the trained Keras model file
        """
        self.model_path = model_path
        self.model = None
        self.test_results = {}
        
    def load_model(self):
        """
        Load the trained model from file.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            print(f"Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully!")
            print("\nModel Architecture:")
            self.model.summary()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def test_on_data(self, data_path=None, test_episodes=5, max_steps=None):
        """
        Test the model on market data.
        
        Args:
            data_path (str): Path to test data JSON file (uses default if None)
            test_episodes (int): Number of test episodes to run
            max_steps (int): Maximum steps per episode (uses default if None)
            
        Returns:
            dict: Dictionary containing test results and metrics
        """
        if self.model is None:
            print("Error: Model not loaded. Call load_model() first.")
            return None
            
        # Use provided data path or default
        data_file = data_path or JSON_FILE_PATH
        max_episode_steps = max_steps or MAX_STEPS_PER_EPISODE
        
        try:
            # Load test data
            print(f"\nLoading test data from: {data_file}")
            df_test = load_data_from_json(data_file)
            print(f"Loaded {len(df_test)} data points for testing")
            
            # Initialize results storage
            episode_results = []
            all_actions = []
            all_rewards = []
            all_portfolio_values = []
            
            print(f"\nRunning {test_episodes} test episodes...")
            
            for episode in range(test_episodes):
                print(f"\n--- Episode {episode + 1}/{test_episodes} ---")
                
                # Create environment for this episode
                env = CryptoTradingEnv(
                    df_test,
                    window_size=WINDOW_SIZE,
                    initial_balance=INITIAL_BALANCE,
                    max_steps=max_episode_steps
                )
                
                # Run single episode
                episode_result = self._run_episode(env, episode + 1)
                episode_results.append(episode_result)
                
                # Collect data for analysis
                all_actions.extend(episode_result['actions'])
                all_rewards.extend(episode_result['rewards'])
                all_portfolio_values.extend(episode_result['portfolio_values'])
            
            # Calculate aggregate metrics
            self.test_results = self._calculate_metrics(
                episode_results, df_test, all_actions, all_rewards, all_portfolio_values
            )
            
            # Print summary
            self._print_summary()
            
            return self.test_results
            
        except Exception as e:
            print(f"Error during testing: {e}")
            return None
    
    def _run_episode(self, env, episode_num):
        """
        Run a single test episode.
        
        Args:
            env: Trading environment
            episode_num (int): Episode number for logging
            
        Returns:
            dict: Episode results including actions, rewards, and portfolio values
        """
        state, info = env.reset()
        
        episode_data = {
            'episode': episode_num,
            'actions': [],
            'rewards': [],
            'portfolio_values': [],
            'prices': [],
            'balances': [],
            'holdings': [],
            'steps': 0
        }
        
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            # Get model prediction
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            q_values = self.model(state_tensor, training=False)
            action = tf.argmax(q_values[0]).numpy()
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record data
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['portfolio_values'].append(info['portfolio_value'])
            episode_data['prices'].append(info['current_price'])
            episode_data['balances'].append(info['balance'])
            episode_data['holdings'].append(info['holdings'])
            
            total_reward += reward
            state = next_state
            step += 1
        
        episode_data['steps'] = step
        episode_data['total_reward'] = total_reward
        episode_data['final_portfolio_value'] = info.get('final_portfolio_value', info['portfolio_value'])
        episode_data['return_pct'] = ((episode_data['final_portfolio_value'] - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
        
        print(f"  Steps: {step}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Final Portfolio Value: {episode_data['final_portfolio_value']:.2f}")
        print(f"  Return: {episode_data['return_pct']:.2f}%")
        
        return episode_data
    
    def _calculate_metrics(self, episode_results, df_test, all_actions, all_rewards, all_portfolio_values):
        """
        Calculate comprehensive performance metrics.
        
        Args:
            episode_results (list): Results from all episodes
            df_test (pd.DataFrame): Test data
            all_actions (list): All actions taken
            all_rewards (list): All rewards received
            all_portfolio_values (list): All portfolio values
            
        Returns:
            dict: Comprehensive metrics dictionary
        """
        # Basic statistics
        returns = [ep['return_pct'] for ep in episode_results]
        final_values = [ep['final_portfolio_value'] for ep in episode_results]
        
        # Calculate buy-and-hold benchmark
        start_price = df_test['close'].iloc[WINDOW_SIZE]
        end_price = df_test['close'].iloc[-1]
        buy_hold_return = ((end_price - start_price) / start_price) * 100
        
        # Action distribution
        action_counts = np.bincount(all_actions, minlength=3)
        action_names = ['Hold', 'Buy', 'Sell']
        
        metrics = {
            'test_summary': {
                'num_episodes': len(episode_results),
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'min_return': np.min(returns),
                'max_return': np.max(returns),
                'avg_final_value': np.mean(final_values),
                'success_rate': len([r for r in returns if r > 0]) / len(returns) * 100
            },
            'benchmark_comparison': {
                'buy_hold_return': buy_hold_return,
                'model_avg_return': np.mean(returns),
                'outperformance': np.mean(returns) - buy_hold_return
            },
            'action_analysis': {
                'action_distribution': {action_names[i]: count for i, count in enumerate(action_counts)},
                'action_percentages': {action_names[i]: (count/len(all_actions))*100 for i, count in enumerate(action_counts)}
            },
            'reward_analysis': {
                'total_rewards': sum(all_rewards),
                'avg_reward': np.mean(all_rewards),
                'reward_std': np.std(all_rewards),
                'positive_rewards': len([r for r in all_rewards if r > 0]),
                'negative_rewards': len([r for r in all_rewards if r < 0])
            },
            'episode_details': episode_results
        }
        
        return metrics
    
    def _print_summary(self):
        """Print a comprehensive summary of test results."""
        if not self.test_results:
            print("No test results available.")
            return
        
        print("\n" + "="*60)
        print("MODEL TESTING SUMMARY")
        print("="*60)
        
        summary = self.test_results['test_summary']
        benchmark = self.test_results['benchmark_comparison']
        actions = self.test_results['action_analysis']
        rewards = self.test_results['reward_analysis']
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Episodes Tested: {summary['num_episodes']}")
        print(f"  Average Return: {summary['avg_return']:.2f}%")
        print(f"  Return Std Dev: {summary['std_return']:.2f}%")
        print(f"  Best Return: {summary['max_return']:.2f}%")
        print(f"  Worst Return: {summary['min_return']:.2f}%")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print(f"  Avg Final Value: ${summary['avg_final_value']:,.2f}")
        
        print(f"\nBENCHMARK COMPARISON:")
        print(f"  Buy & Hold Return: {benchmark['buy_hold_return']:.2f}%")
        print(f"  Model Avg Return: {benchmark['model_avg_return']:.2f}%")
        print(f"  Outperformance: {benchmark['outperformance']:.2f}%")
        
        print(f"\nTRADING BEHAVIOR:")
        for action, percentage in actions['action_percentages'].items():
            count = actions['action_distribution'][action]
            print(f"  {action}: {count:,} times ({percentage:.1f}%)")
        
        print(f"\nREWARD ANALYSIS:")
        print(f"  Total Rewards: {rewards['total_rewards']:.2f}")
        print(f"  Average Reward: {rewards['avg_reward']:.4f}")
        print(f"  Reward Std Dev: {rewards['reward_std']:.4f}")
        print(f"  Positive Rewards: {rewards['positive_rewards']:,}")
        print(f"  Negative Rewards: {rewards['negative_rewards']:,}")
    
    def save_results(self, filename=None):
        """
        Save test results to a JSON file.
        
        Args:
            filename (str): Output filename (auto-generated if None)
        """
        if not self.test_results:
            print("No test results to save.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_test_results_{timestamp}.json"
        
        try:
            # Convert numpy types to Python types for JSON serialization
            results_copy = self._convert_for_json(self.test_results)
            
            with open(filename, 'w') as f:
                json.dump(results_copy, f, indent=2)
            
            print(f"\nTest results saved to: {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def plot_results(self, save_plots=True):
        """
        Generate visualization plots of the test results.
        
        Args:
            save_plots (bool): Whether to save plots to files
        """
        if not self.test_results:
            print("No test results to plot.")
            return
        
        try:
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Model Testing Results', fontsize=16)
            
            # Plot 1: Episode Returns
            episodes = range(1, len(self.test_results['episode_details']) + 1)
            returns = [ep['return_pct'] for ep in self.test_results['episode_details']]
            
            axes[0, 0].bar(episodes, returns, alpha=0.7, color='blue')
            axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 0].axhline(y=self.test_results['benchmark_comparison']['buy_hold_return'], 
                              color='green', linestyle='--', alpha=0.7, label='Buy & Hold')
            axes[0, 0].set_title('Returns by Episode')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Return (%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Action Distribution
            actions = self.test_results['action_analysis']['action_distribution']
            action_names = list(actions.keys())
            action_counts = list(actions.values())
            
            axes[0, 1].pie(action_counts, labels=action_names, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Action Distribution')
            
            # Plot 3: Portfolio Value Evolution (first episode)
            if self.test_results['episode_details']:
                first_episode = self.test_results['episode_details'][0]
                steps = range(len(first_episode['portfolio_values']))
                
                axes[1, 0].plot(steps, first_episode['portfolio_values'], 'b-', linewidth=2, label='Portfolio Value')
                axes[1, 0].axhline(y=INITIAL_BALANCE, color='red', linestyle='--', alpha=0.7, label='Initial Balance')
                axes[1, 0].set_title('Portfolio Value Evolution (Episode 1)')
                axes[1, 0].set_xlabel('Steps')
                axes[1, 0].set_ylabel('Portfolio Value ($)')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Return Distribution
            axes[1, 1].hist(returns, bins=10, alpha=0.7, color='green', edgecolor='black')
            axes[1, 1].axvline(x=np.mean(returns), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(returns):.2f}%')
            axes[1, 1].axvline(x=self.test_results['benchmark_comparison']['buy_hold_return'], 
                              color='blue', linestyle='--', label='Buy & Hold')
            axes[1, 1].set_title('Return Distribution')
            axes[1, 1].set_xlabel('Return (%)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = f"model_test_plots_{timestamp}.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                print(f"Plots saved to: {plot_filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating plots: {e}")


def main():
    """
    Main function to run model testing.
    """
    print("Cryptocurrency Trading Bot - Model Tester")
    print("=" * 50)
    
    # Initialize tester
    tester = ModelTester()
    
    # Load model
    if not tester.load_model():
        print("Failed to load model. Exiting.")
        return
    
    # Run tests
    print("\nStarting model evaluation...")
    results = tester.test_on_data(test_episodes=3, max_steps=500)
    
    if results:
        # Save results
        tester.save_results()
        
        # Generate plots
        print("\nGenerating visualization plots...")
        tester.plot_results()
        
        print("\nModel testing completed successfully!")
    else:
        print("Model testing failed.")


if __name__ == "__main__":
    main()