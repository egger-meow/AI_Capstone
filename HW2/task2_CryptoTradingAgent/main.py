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
from dqn_agent import create_agent, train_agent, evaluate_agent, save_agent, load_agent, TensorboardCallback
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


def train_crypto_agent(args):
    """
    Train a cryptocurrency trading agent
    
    Args:
        args: Command-line arguments
    """
    print_header("TRAINING CRYPTOCURRENCY TRADING AGENT")
    
    # Create directories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Prepare data
    data_path = f"data/{args.symbol.replace('-', '_')}_{args.interval}.csv"
    df = prepare_data_for_env(
        data_path=data_path,
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Split data
    train_df, val_df, test_df = split_data(df, args.train_ratio, args.val_ratio, args.test_ratio)
    
    print(f"Data prepared successfully:")
    print(f"  Total data points: {len(df)}")
    print(f"  Training data points: {len(train_df)}")
    print(f"  Validation data points: {len(val_df)}")
    print(f"  Test data points: {len(test_df)}")
    
    # Create training environment
    env = CryptoTradingEnv(
        df=train_df,
        window_size=args.window_size,
        commission=args.commission,
        initial_balance=args.initial_balance,
        reward_scaling=args.reward_scaling
    )
    
    # Wrap environment
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Create agent
    model = create_agent(
        env=env,
        tensorboard_log="./logs/",
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        exploration_fraction=args.exploration_fraction
    )
    
    # Train agent
    print("\nTraining agent...")
    start_time = time.time()
    model = train_agent(model, args.timesteps)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save model
    model_path = f"models/{args.symbol.replace('-', '_')}_dqn_{args.timesteps}"
    save_agent(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate on validation data
    print("\nEvaluating on validation data...")
    val_env = CryptoTradingEnv(
        df=val_df,
        window_size=args.window_size,
        commission=args.commission,
        initial_balance=args.initial_balance,
        reward_scaling=args.reward_scaling
    )
    
    # Evaluate agent
    rewards, infos = evaluate_agent(model, val_env, num_episodes=1)
    
    print(f"Validation performance:")
    print(f"  Average reward: {np.mean(rewards):.2f}")
    print(f"  Final portfolio value: ${infos[-1][-1]['net_worth']:.2f}")
    
    # Get transaction history
    tx_history = val_env.get_transaction_history()
    
    # Plot results if requested
    if args.visualize:
        # Plot price history
        plot_price_history(
            val_df,
            title=f"{args.symbol} Price History (Validation Set)",
            save_path=f"results/{args.symbol.replace('-', '_')}_price_history.png"
        )
        
        # Plot trading actions
        if not tx_history.empty:
            plot_trading_actions(
                val_df,
                tx_history,
                title=f"Trading Actions for {args.symbol}",
                save_path=f"results/{args.symbol.replace('-', '_')}_trading_actions.png"
            )
        
        # Plot portfolio performance
        if not tx_history.empty:
            metrics = plot_portfolio_performance(
                tx_history,
                val_df,
                initial_balance=args.initial_balance,
                commission=args.commission,
                title=f"Portfolio Performance for {args.symbol}",
                save_path=f"results/{args.symbol.replace('-', '_')}_portfolio_performance.png"
            )
            
            # Plot performance metrics
            plot_performance_metrics(
                metrics,
                title=f"Performance Metrics for {args.symbol}",
                save_path=f"results/{args.symbol.replace('-', '_')}_performance_metrics.png"
            )
        
        # Create animation
        if not tx_history.empty and args.create_animation:
            create_trading_animation(
                val_df,
                tx_history,
                title=f"Trading Animation for {args.symbol}",
                save_path=f"results/{args.symbol.replace('-', '_')}_trading_animation.mp4"
            )
    
    print("\nTraining and evaluation completed successfully!")


def test_crypto_agent(args):
    """
    Test a trained cryptocurrency trading agent
    
    Args:
        args: Command-line arguments
    """
    print_header("TESTING CRYPTOCURRENCY TRADING AGENT")
    
    # Load data
    data_path = f"data/{args.symbol.replace('-', '_')}_{args.interval}.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found. Downloading data...")
        df = prepare_data_for_env(
            data_path=data_path,
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date
        )
    else:
        df = prepare_data_for_env(data_path=data_path)
    
    # Split data
    _, _, test_df = split_data(df, args.train_ratio, args.val_ratio, args.test_ratio)
    
    print(f"Test data loaded: {len(test_df)} data points")
    
    # Create test environment
    test_env = CryptoTradingEnv(
        df=test_df,
        window_size=args.window_size,
        commission=args.commission,
        initial_balance=args.initial_balance,
        reward_scaling=args.reward_scaling
    )
    
    # Load model
    model_path = f"models/{args.symbol.replace('-', '_')}_dqn_{args.timesteps}"
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}. Please train a model first.")
        return
    
    model = load_agent(model_path)
    print(f"Model loaded from {model_path}")
    
    # Evaluate agent
    print("\nEvaluating on test data...")
    rewards, infos = evaluate_agent(model, test_env, num_episodes=1)
    
    print(f"Test performance:")
    print(f"  Average reward: {np.mean(rewards):.2f}")
    print(f"  Final portfolio value: ${infos[-1][-1]['net_worth']:.2f}")
    
    # Get transaction history
    tx_history = test_env.get_transaction_history()
    
    # Plot results if requested
    if args.visualize:
        # Plot price history
        plot_price_history(
            test_df,
            title=f"{args.symbol} Price History (Test Set)",
            save_path=f"results/{args.symbol.replace('-', '_')}_test_price_history.png"
        )
        
        # Plot trading actions
        if not tx_history.empty:
            plot_trading_actions(
                test_df,
                tx_history,
                title=f"Trading Actions for {args.symbol} (Test Set)",
                save_path=f"results/{args.symbol.replace('-', '_')}_test_trading_actions.png"
            )
        
        # Plot portfolio performance
        if not tx_history.empty:
            metrics = plot_portfolio_performance(
                tx_history,
                test_df,
                initial_balance=args.initial_balance,
                commission=args.commission,
                title=f"Portfolio Performance for {args.symbol} (Test Set)",
                save_path=f"results/{args.symbol.replace('-', '_')}_test_portfolio_performance.png"
            )
            
            # Plot performance metrics
            plot_performance_metrics(
                metrics,
                title=f"Performance Metrics for {args.symbol} (Test Set)",
                save_path=f"results/{args.symbol.replace('-', '_')}_test_performance_metrics.png"
            )
        
        # Create animation
        if not tx_history.empty and args.create_animation:
            create_trading_animation(
                test_df,
                tx_history,
                title=f"Trading Animation for {args.symbol} (Test Set)",
                save_path=f"results/{args.symbol.replace('-', '_')}_test_trading_animation.mp4"
            )
    
    print("\nTesting completed successfully!")


def visualize_crypto_data(args):
    """
    Visualize cryptocurrency data
    
    Args:
        args: Command-line arguments
    """
    print_header("VISUALIZING CRYPTOCURRENCY DATA")
    
    # Load data
    data_path = f"data/{args.symbol.replace('-', '_')}_{args.interval}.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found. Downloading data...")
        df = prepare_data_for_env(
            data_path=data_path,
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date
        )
    else:
        df = prepare_data_for_env(data_path=data_path)
    
    print(f"Data loaded: {len(df)} data points")
    
    # Create directories if they don't exist
    os.makedirs("results", exist_ok=True)
    
    # Plot price history
    plot_price_history(
        df,
        title=f"{args.symbol} Price History",
        save_path=f"results/{args.symbol.replace('-', '_')}_full_price_history.png"
    )
    
    # Plot with technical indicators
    if args.show_indicators:
        indicators = ['Close', 'SMA_7', 'SMA_20', 'EMA_7', 'EMA_20']
        available_columns = [col for col in indicators if col in df.columns]
        
        plot_price_history(
            df,
            columns=available_columns,
            title=f"{args.symbol} Price with Moving Averages",
            save_path=f"results/{args.symbol.replace('-', '_')}_with_ma.png"
        )
        
        # Plot RSI if available
        if 'RSI' in df.columns:
            plt.figure(figsize=(12, 4))
            plt.plot(df.index, df['RSI'])
            plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            plt.title(f"{args.symbol} RSI")
            plt.ylabel('RSI')
            plt.tight_layout()
            plt.savefig(f"results/{args.symbol.replace('-', '_')}_rsi.png")
            plt.show()
        
        # Plot MACD if available
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            plt.figure(figsize=(12, 4))
            plt.plot(df.index, df['MACD'], label='MACD')
            plt.plot(df.index, df['MACD_Signal'], label='Signal Line')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            plt.title(f"{args.symbol} MACD")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"results/{args.symbol.replace('-', '_')}_macd.png")
            plt.show()
    
    print("\nVisualization completed successfully!")


def compare_strategies(args):
    """
    Compare different trading strategies
    
    Args:
        args: Command-line arguments
    """
    print_header("COMPARING TRADING STRATEGIES")
    
    # Load data
    data_path = f"data/{args.symbol.replace('-', '_')}_{args.interval}.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found. Downloading data...")
        df = prepare_data_for_env(
            data_path=data_path,
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date
        )
    else:
        df = prepare_data_for_env(data_path=data_path)
    
    # Split data
    _, _, test_df = split_data(df, args.train_ratio, args.val_ratio, args.test_ratio)
    
    print(f"Data loaded: {len(test_df)} data points for testing")
    
    # Implement simple strategies for comparison
    
    # 1. Buy and Hold Strategy
    def buy_and_hold(df, initial_balance, commission):
        # Buy at the beginning and hold
        price_start = df['Close'].iloc[0]
        price_end = df['Close'].iloc[-1]
        
        # Calculate returns
        crypto_bought = initial_balance / price_start * (1 - commission)
        final_value = crypto_bought * price_end
        return_pct = (final_value / initial_balance - 1) * 100
        
        print(f"Buy and Hold Strategy:")
        print(f"  Initial Investment: ${initial_balance:.2f}")
        print(f"  Final Value: ${final_value:.2f}")
        print(f"  Return: {return_pct:.2f}%")
        
        return {
            'strategy': 'Buy and Hold',
            'initial_balance': initial_balance,
            'final_value': final_value,
            'return_pct': return_pct
        }
    
    # 2. SMA Crossover Strategy
    def sma_crossover(df, initial_balance, commission):
        df = df.copy()
        
        # Calculate SMAs if not present
        if 'SMA_7' not in df.columns:
            df['SMA_7'] = df['Close'].rolling(window=7).mean()
        if 'SMA_20' not in df.columns:
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Drop NaN values
        df = df.dropna()
        
        # Generate signals
        df['signal'] = 0
        df['signal'][7:] = np.where(df['SMA_7'][7:] > df['SMA_20'][7:], 1, 0)
        df['position'] = df['signal'].diff()
        
        # Initialize variables
        balance = initial_balance
        crypto_held = 0
        transactions = []
        
        # Execute trades
        for i in range(1, len(df)):
            if df['position'].iloc[i] == 1:  # Buy signal
                if balance > 0:
                    price = df['Close'].iloc[i]
                    crypto_bought = balance / price * (1 - commission)
                    crypto_held += crypto_bought
                    transactions.append({
                        'step': i,
                        'type': 'buy',
                        'price': price,
                        'amount': crypto_bought,
                        'balance': 0,
                        'crypto_held': crypto_held
                    })
                    balance = 0
            
            elif df['position'].iloc[i] == -1:  # Sell signal
                if crypto_held > 0:
                    price = df['Close'].iloc[i]
                    balance += crypto_held * price * (1 - commission)
                    transactions.append({
                        'step': i,
                        'type': 'sell',
                        'price': price,
                        'amount': crypto_held,
                        'balance': balance,
                        'crypto_held': 0
                    })
                    crypto_held = 0
        
        # Final evaluation
        final_value = balance + crypto_held * df['Close'].iloc[-1]
        return_pct = (final_value / initial_balance - 1) * 100
        
        print(f"SMA Crossover Strategy:")
        print(f"  Initial Investment: ${initial_balance:.2f}")
        print(f"  Final Value: ${final_value:.2f}")
        print(f"  Return: {return_pct:.2f}%")
        print(f"  Number of Trades: {len(transactions)}")
        
        return {
            'strategy': 'SMA Crossover',
            'initial_balance': initial_balance,
            'final_value': final_value,
            'return_pct': return_pct,
            'transactions': pd.DataFrame(transactions)
        }
    
    # Execute baseline strategies
    buy_hold_results = buy_and_hold(test_df, args.initial_balance, args.commission)
    sma_results = sma_crossover(test_df, args.initial_balance, args.commission)
    
    # Run DQN agent
    test_env = CryptoTradingEnv(
        df=test_df,
        window_size=args.window_size,
        commission=args.commission,
        initial_balance=args.initial_balance,
        reward_scaling=args.reward_scaling
    )
    
    # Load model
    model_path = f"models/{args.symbol.replace('-', '_')}_dqn_{args.timesteps}"
    if not os.path.exists(model_path + ".zip"):
        print(f"DQN model not found at {model_path}. Skipping DQN comparison.")
        dqn_results = None
    else:
        model = load_agent(model_path)
        print(f"DQN model loaded from {model_path}")
        
        # Evaluate agent
        rewards, infos = evaluate_agent(model, test_env, num_episodes=1)
        final_value = infos[-1][-1]['net_worth']
        return_pct = (final_value / args.initial_balance - 1) * 100
        
        print(f"DQN Agent Strategy:")
        print(f"  Initial Investment: ${args.initial_balance:.2f}")
        print(f"  Final Value: ${final_value:.2f}")
        print(f"  Return: {return_pct:.2f}%")
        
        # Get transaction history
        tx_history = test_env.get_transaction_history()
        print(f"  Number of Trades: {len(tx_history)}")
        
        dqn_results = {
            'strategy': 'DQN Agent',
            'initial_balance': args.initial_balance,
            'final_value': final_value,
            'return_pct': return_pct,
            'transactions': tx_history
        }
    
    # Plot comparison
    if args.visualize:
        # Create directories if they don't exist
        os.makedirs("results", exist_ok=True)
        
        # Compare returns
        strategies = []
        returns = []
        
        strategies.append(buy_hold_results['strategy'])
        returns.append(buy_hold_results['return_pct'])
        
        strategies.append(sma_results['strategy'])
        returns.append(sma_results['return_pct'])
        
        if dqn_results is not None:
            strategies.append(dqn_results['strategy'])
            returns.append(dqn_results['return_pct'])
        
        # Plot bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(strategies, returns)
        
        # Color bars based on positive/negative returns
        for i, bar in enumerate(bars):
            if returns[i] < 0:
                bar.set_color('red')
            else:
                bar.set_color('green')
        
        # Add value labels
        for i, v in enumerate(returns):
            plt.text(i, v + (1 if v >= 0 else -1), 
                    f"{v:.2f}%", ha='center', va='bottom' if v >= 0 else 'top')
        
        plt.title(f"Performance Comparison ({args.symbol})")
        plt.ylabel('Return (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"results/{args.symbol.replace('-', '_')}_strategy_comparison.png")
        plt.show()
        
        # Plot trading actions for SMA strategy
        if not sma_results['transactions'].empty:
            plot_trading_actions(
                test_df,
                sma_results['transactions'],
                title=f"SMA Crossover Trading Actions for {args.symbol}",
                save_path=f"results/{args.symbol.replace('-', '_')}_sma_trading_actions.png"
            )
        
        # Plot trading actions for DQN strategy
        if dqn_results is not None and not dqn_results['transactions'].empty:
            plot_trading_actions(
                test_df,
                dqn_results['transactions'],
                title=f"DQN Agent Trading Actions for {args.symbol}",
                save_path=f"results/{args.symbol.replace('-', '_')}_dqn_trading_actions.png"
            )
    
    print("\nStrategy comparison completed successfully!")


def check_environment(args):
    """
    Check if the environment is working correctly
    
    Args:
        args: Command-line arguments
    """
    print_header("CHECKING ENVIRONMENT")
    
    # Create a small dataframe for testing
    df = pd.DataFrame({
        'Open': [100, 101, 102, 103, 104],
        'High': [105, 106, 107, 108, 109],
        'Low': [95, 96, 97, 98, 99],
        'Close': [101, 102, 103, 104, 105],
        'Volume': [1000, 1100, 1200, 1300, 1400]
    })
    df.index = pd.date_range(start='2023-01-01', periods=len(df))
    
    # Create the environment
    env = CryptoTradingEnv(
        df=df,
        window_size=3,
        initial_balance=10000,
        commission=0.001
    )
    
    # Reset the environment
    obs, _ = env.reset()
    
    print("Environment initialized successfully!")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    
    print("\nTook a random action:")
    print(f"  Action: {action} ({['HOLD', 'BUY', 'SELL'][action]})")
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    print(f"  Info: {info}")
    
    print("\nEnvironment check completed successfully!")


def main():
    """
    Main function to parse arguments and run the appropriate command
    """
    parser = argparse.ArgumentParser(description="Cryptocurrency Trading Agent")
    parser.add_argument('command', type=str, choices=['train', 'test', 'visualize', 'compare', 'check'],
                      help='Command to run')
    
    # Data parameters
    parser.add_argument('--symbol', type=str, default="BTC-USD",
                      help='Cryptocurrency symbol (default: BTC-USD)')
    parser.add_argument('--interval', type=str, default="1h",
                      help='Data interval (default: 1h)')
    parser.add_argument('--start-date', type=str, default=None,
                      help='Start date for data (format: YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                      help='End date for data (format: YYYY-MM-DD)')
    
    # Environment parameters
    parser.add_argument('--window-size', type=int, default=20,
                      help='Window size for observations (default: 20)')
    parser.add_argument('--initial-balance', type=float, default=10000,
                      help='Initial balance (default: 10000)')
    parser.add_argument('--commission', type=float, default=0.001,
                      help='Trading commission (default: 0.001)')
    parser.add_argument('--reward-scaling', type=float, default=1.0,
                      help='Reward scaling factor (default: 1.0)')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=10000,
                      help='Number of timesteps to train for (default: 10000)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                      help='Learning rate (default: 0.0001)')
    parser.add_argument('--buffer-size', type=int, default=10000,
                      help='Replay buffer size (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size (default: 64)')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor (default: 0.99)')
    parser.add_argument('--exploration-fraction', type=float, default=0.1,
                      help='Exploration fraction (default: 0.1)')
    
    # Data split parameters
    parser.add_argument('--train-ratio', type=float, default=0.7,
                      help='Training data ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                      help='Validation data ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                      help='Test data ratio (default: 0.15)')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize results')
    parser.add_argument('--show-indicators', action='store_true',
                      help='Show technical indicators in visualizations')
    parser.add_argument('--create-animation', action='store_true',
                      help='Create trading animation')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Start timer
    start_time = time.time()
    
    # Run the appropriate command
    if args.command == 'train':
        train_crypto_agent(args)
    elif args.command == 'test':
        test_crypto_agent(args)
    elif args.command == 'visualize':
        visualize_crypto_data(args)
    elif args.command == 'compare':
        compare_strategies(args)
    elif args.command == 'check':
        check_environment(args)
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main() 