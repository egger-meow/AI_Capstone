# Cryptocurrency Trading Agent with Reinforcement Learning

This project implements a simple but effective reinforcement learning agent for cryptocurrency trading. The agent learns to make trading decisions (buy, sell, hold) based on historical price data to maximize profit.

## Project Structure

```
task2_CryptoTradingAgent/
├── README.md                    # This file
├── crypto_trading_env.py        # Custom trading environment
├── dqn_agent.py                 # DQN agent implementation
├── main.py                      # Main script to run experiments
├── data_utils.py                # Utilities for data loading/processing
├── visualization.py             # Visualization tools for results
├── examples/                    # Example notebooks/scripts
│   └── basic_trading_example.py # Simple example to get started
└── data/                        # Directory for price data
    └── BTC_USD_1h.csv           # Sample price data (will be downloaded)
```

## Features

- **Simple Trading Environment**: Custom OpenAI Gym environment for cryptocurrency trading
- **DQN Agent**: Deep Q-Network implementation for trading decisions
- **Data Processing**: Tools to download and process historical cryptocurrency data
- **Visualization**: Tools to visualize trading performance and agent behavior
- **Configurable**: Easy to modify parameters and strategies

## Requirements

- Python 3.8+
- Gymnasium
- Stable-Baselines3
- Pandas
- NumPy
- Matplotlib
- yfinance (for data download)

## Installation

1. Install dependencies:
   ```
   pip install gymnasium stable-baselines3 pandas numpy matplotlib yfinance
   ```

## Quick Start

To get started with a simple example:

```python
# Navigate to the examples directory
cd task2_CryptoTradingAgent/examples

# Run the basic example
python basic_trading_example.py
```

This will download some Bitcoin data, train a simple DQN agent, and visualize the results.

## Using the Main Script

The `main.py` script provides a command-line interface for running experiments:

1. Check that the environment works correctly:
   ```
   python main.py check
   ```

2. Train a trading agent:
   ```
   python main.py train --symbol BTC-USD --timesteps 10000 --visualize
   ```

3. Test a trained agent:
   ```
   python main.py test --symbol BTC-USD --timesteps 10000 --visualize
   ```

4. Visualize crypto data:
   ```
   python main.py visualize --symbol BTC-USD --show-indicators
   ```

5. Compare different strategies:
   ```
   python main.py compare --symbol BTC-USD --visualize
   ```

## Available Command-Line Options

- `--symbol`: Cryptocurrency symbol (default: "BTC-USD")
- `--interval`: Data interval (default: "1h")
- `--start-date`: Start date for data (format: YYYY-MM-DD)
- `--end-date`: End date for data (format: YYYY-MM-DD)
- `--window-size`: Window size for observations (default: 20)
- `--initial-balance`: Initial balance (default: 10000)
- `--commission`: Trading commission (default: 0.001)
- `--reward-scaling`: Reward scaling factor (default: 1.0)
- `--timesteps`: Number of timesteps to train for (default: 10000)
- `--learning-rate`: Learning rate (default: 0.0001)
- `--buffer-size`: Replay buffer size (default: 10000)
- `--batch-size`: Batch size (default: 64)
- `--gamma`: Discount factor (default: 0.99)
- `--exploration-fraction`: Exploration fraction (default: 0.1)
- `--visualize`: Visualize results (flag)
- `--show-indicators`: Show technical indicators in visualizations (flag)
- `--create-animation`: Create trading animation (flag)

## Extending the Project

### Adding New Technical Indicators

You can add new technical indicators in the `data_utils.py` file:

```python
def add_technical_indicators(df):
    # ... existing code ...
    
    # Add your custom indicator
    df['my_indicator'] = ...
    
    return df
```

### Modifying the Reward Function

To change how the agent is rewarded, modify the `_calculate_reward` method in `crypto_trading_env.py`:

```python
def _calculate_reward(self, action):
    # ... your custom reward logic ...
    return reward
```

### Creating Custom Trading Strategies

You can implement custom strategies in the `compare_strategies` function in `main.py`:

```python
def my_custom_strategy(df, initial_balance, commission):
    # ... your strategy logic ...
    return {
        'strategy': 'My Custom Strategy',
        'initial_balance': initial_balance,
        'final_value': final_value,
        'return_pct': return_pct,
        'transactions': pd.DataFrame(transactions)
    }
```

## Report Guidance

When writing your report, consider analyzing:

1. How different parameters affect agent performance
2. Comparison with simple baseline strategies (buy and hold, etc.)
3. Analysis of when/why the agent makes trading decisions
4. Risk-adjusted performance metrics
5. Limitations and potential improvements

## References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [FinRL: Financial Reinforcement Learning](https://github.com/AI4Finance-Foundation/FinRL)
- [Gym-Anytrading](https://github.com/AminHP/gym-anytrading) 