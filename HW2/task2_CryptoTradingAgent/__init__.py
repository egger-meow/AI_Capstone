"""
Cryptocurrency Trading Agent with Reinforcement Learning.

A simple but effective reinforcement learning framework for cryptocurrency trading.
"""

__version__ = "0.1.0"
__author__ = "AI Capstone Project"

# Export main classes and functions
from .crypto_trading_env import CryptoTradingEnv, Actions
from .dqn_agent import create_agent, train_agent, evaluate_agent, save_agent, load_agent
from .data_utils import prepare_data_for_env, add_technical_indicators, split_data
from .visualization import (plot_price_history, plot_trading_actions, 
                          plot_portfolio_performance, create_trading_animation,
                          plot_performance_metrics) 