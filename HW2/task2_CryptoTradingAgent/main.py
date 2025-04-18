import os
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Import project modules
from crypto_trading_env import CryptoTradingEnv
from trading_agent import TradingAgent, TradingCallback
from data_utils import prepare_data_for_env, split_data
from visualization import (plot_price_history, plot_trading_actions, 
                         plot_portfolio_performance, create_trading_animation,
                         plot_performance_metrics)


def print_header(title):
    """
    Print a formatted header
    """
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")


def train(args):
    # Prepare data
    print("Preparing data...")
    df = prepare_data_for_env(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        add_indicators=True,
        normalize=True
    )
    
    # Split data
    train_df, val_df, test_df = split_data(df)
    
    # Create environments
    train_env = CryptoTradingEnv(train_df, initial_balance=args.initial_balance)
    val_env = CryptoTradingEnv(val_df, initial_balance=args.initial_balance)
    test_env = CryptoTradingEnv(test_df, initial_balance=args.initial_balance)
    
    # Wrap environments
    train_env = Monitor(train_env)
    val_env = Monitor(val_env)
    test_env = Monitor(test_env)
    
    train_env = DummyVecEnv([lambda: train_env])
    val_env = DummyVecEnv([lambda: val_env])
    test_env = DummyVecEnv([lambda: test_env])
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Create callback
    callback = TradingCallback(
        check_freq=args.check_freq,
        log_dir=args.log_dir,
        verbose=1
    )
    
    # Create and train agent
    print(f"Training {args.algorithm.upper()} agent...")
    agent = TradingAgent(
        train_env,
        algorithm=args.algorithm,
        use_her=args.use_her,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        tensorboard_log=args.log_dir
    )
    
    agent.train(
        total_timesteps=args.total_timesteps,
        callback=callback
    )
    
    # Save final model
    agent.save(os.path.join(args.model_dir, "final_model"))
    
    # Evaluate on test set
    print("Evaluating on test set...")
    mean_reward, std_reward = agent.evaluate(test_env)
    print(f"Test mean reward: {mean_reward:.2f} ± {std_reward:.2f}")


def evaluate(args):
    # Load data
    print("Loading data...")
    df = prepare_data_for_env(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        add_indicators=True,
        normalize=True
    )
    
    # Create environment
    env = CryptoTradingEnv(df, initial_balance=args.initial_balance)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Load agent
    print(f"Loading {args.algorithm.upper()} agent...")
    agent = TradingAgent(env, algorithm=args.algorithm, use_her=args.use_her)
    agent.load(args.model_path)
    
    # Evaluate
    print("Evaluating...")
    mean_reward, std_reward = agent.evaluate(env)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")


def main():
    """
    Main function to parse arguments and run the appropriate command
    """
    parser = argparse.ArgumentParser(description="Crypto Trading Agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the agent")
    train_parser.add_argument("--data_path", type=str, default="data/btc-usd-1h.csv")
    train_parser.add_argument("--symbol", type=str, default="BTC-USD")
    train_parser.add_argument("--start_date", type=str, default=None)
    train_parser.add_argument("--end_date", type=str, default=None)
    train_parser.add_argument("--interval", type=str, default="1h")
    train_parser.add_argument("--initial_balance", type=float, default=10000.0)
    train_parser.add_argument("--algorithm", type=str, default="sac", choices=["sac", "td3", "ddpg"])
    train_parser.add_argument("--use_her", action="store_true")
    train_parser.add_argument("--learning_rate", type=float, default=0.0003)
    train_parser.add_argument("--buffer_size", type=int, default=1000000)
    train_parser.add_argument("--batch_size", type=int, default=256)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--tau", type=float, default=0.005)
    train_parser.add_argument("--total_timesteps", type=int, default=100000)
    train_parser.add_argument("--check_freq", type=int, default=1000)
    train_parser.add_argument("--model_dir", type=str, default="models")
    train_parser.add_argument("--log_dir", type=str, default="logs")
    train_parser.set_defaults(func=train)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the agent")
    eval_parser.add_argument("--data_path", type=str, default="data/btc-usd-1h.csv")
    eval_parser.add_argument("--symbol", type=str, default="BTC-USD")
    eval_parser.add_argument("--start_date", type=str, default=None)
    eval_parser.add_argument("--end_date", type=str, default=None)
    eval_parser.add_argument("--interval", type=str, default="1h")
    eval_parser.add_argument("--initial_balance", type=float, default=10000.0)
    eval_parser.add_argument("--algorithm", type=str, default="sac", choices=["sac", "td3", "ddpg"])
    eval_parser.add_argument("--use_her", action="store_true")
    eval_parser.add_argument("--model_path", type=str, required=True)
    eval_parser.set_defaults(func=evaluate)
    
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 