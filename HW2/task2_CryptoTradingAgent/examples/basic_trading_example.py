#!/usr/bin/env python
"""
Basic example of how to use the cryptocurrency trading agent.
This script demonstrates a simple workflow for training and evaluating the agent.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from crypto_trading_env import CryptoTradingEnv, Actions
from dqn_agent import create_agent, train_agent, evaluate_agent, save_agent, load_agent
from data_utils import prepare_data_for_env, split_data
from visualization import plot_price_history, plot_trading_actions, plot_portfolio_performance


def main():
    print("=== Cryptocurrency Trading Agent Basic Example ===")
    
    # Step 1: Create directories for outputs
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../examples/outputs", exist_ok=True)
    
    # Step 2: Download and prepare Bitcoin data
    print("\nStep 1: Downloading and preparing data...")
    symbol = "BTC-USD"
    interval = "1h"
    
    # Use recent data (7 days for this example)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # 7 days for a quick example
    
    data_path = f"../data/{symbol.replace('-', '_')}_{interval}_example.csv"
    
    # Check if we already have data
    if os.path.exists(data_path):
        print(f"Loading existing data from {data_path}")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        print(f"Downloading {symbol} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        df = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval=interval)
        df.to_csv(data_path)
    
    # Add technical indicators
    from data_utils import add_technical_indicators
    df = add_technical_indicators(df)
    
    # Split data for training and evaluation
    train_df, val_df, test_df = split_data(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    print(f"Data prepared successfully: {len(df)} data points")
    print(f"  Training data: {len(train_df)} points")
    print(f"  Validation data: {len(val_df)} points")
    print(f"  Test data: {len(test_df)} points")
    
    # Step 3: Create and configure the trading environment
    print("\nStep 2: Creating trading environment...")
    
    env_params = {
        "window_size": 20,
        "initial_balance": 10000,
        "commission": 0.001,
        "reward_scaling": 1.0
    }
    
    # Create environment
    env = CryptoTradingEnv(
        df=train_df,
        **env_params
    )
    
    # Step 4: Create and train the DQN agent
    print("\nStep 3: Training the DQN agent...")
    
    # The timesteps here are very small for quick demonstration purposes only
    # For real training, use at least 50,000 - 100,000 timesteps
    timesteps = 1000  # Very small for demonstration
    
    dqn_params = {
        "learning_rate": 0.0001,
        "buffer_size": 10000,
        "batch_size": 64,
        "gamma": 0.99,
        "exploration_fraction": 0.2,  # Higher exploration for quick learning
    }
    
    # Create agent
    model = create_agent(env=env, **dqn_params)
    
    # Train agent (this will take some time)
    model = train_agent(model, timesteps=timesteps, progress_bar=True)
    
    # Save the agent
    model_path = "../examples/outputs/example_model"
    save_agent(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Step 5: Evaluate the agent
    print("\nStep 4: Evaluating the agent...")
    
    # Create evaluation environment
    eval_env = CryptoTradingEnv(
        df=test_df,
        **env_params
    )
    
    # Evaluate agent
    rewards, infos = evaluate_agent(model, eval_env, num_episodes=1)
    
    print(f"Evaluation results:")
    print(f"  Average reward: {np.mean(rewards):.2f}")
    print(f"  Final portfolio value: ${infos[-1][-1]['net_worth']:.2f}")
    
    # Get transaction history
    tx_history = eval_env.get_transaction_history()
    
    # Step 6: Visualize results
    print("\nStep 5: Visualizing results...")
    
    # Plot price history
    plot_price_history(
        test_df,
        title=f"{symbol} Price History (Test Set)",
        save_path="../examples/outputs/price_history.png"
    )
    
    # Plot trading actions
    if not tx_history.empty:
        plot_trading_actions(
            test_df,
            tx_history,
            title=f"Trading Actions for {symbol}",
            save_path="../examples/outputs/trading_actions.png"
        )
    
    # Plot portfolio performance
    if not tx_history.empty:
        metrics = plot_portfolio_performance(
            tx_history,
            test_df,
            initial_balance=env_params["initial_balance"],
            commission=env_params["commission"],
            title=f"Portfolio Performance for {symbol}",
            save_path="../examples/outputs/portfolio_performance.png"
        )
        
        print("\nPerformance metrics:")
        print(f"  Final portfolio value: ${metrics['final_portfolio_value']:.2f}")
        print(f"  Return: {metrics['final_return_pct']:.2f}%")
        print(f"  Buy & Hold return: {metrics['buy_hold_return_pct']:.2f}%")
        print(f"  Outperformance vs Buy & Hold: {metrics['outperformance_pct']:.2f}%")
    
    print("\n=== Example completed successfully! ===")
    print("Check the 'examples/outputs' directory for results.")


if __name__ == "__main__":
    main() 