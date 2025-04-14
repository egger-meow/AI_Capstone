import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")


def plot_price_history(df, start_idx=None, end_idx=None, 
                      columns=['Close'], title='Cryptocurrency Price History',
                      figsize=(12, 6), save_path=None):
    """
    Plot price history for cryptocurrency
    
    Args:
        df (pandas.DataFrame): Price data with DateTimeIndex
        start_idx (int): Starting index for plotting
        end_idx (int): Ending index for plotting
        columns (list): List of columns to plot from DataFrame
        title (str): Title for the plot
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Slice dataframe if indices provided
    if start_idx is not None or end_idx is not None:
        df = df.iloc[slice(start_idx, end_idx)]
    
    # Plot selected columns
    for col in columns:
        if col in df.columns:
            plt.plot(df.index, df[col], label=col)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_trading_actions(df, transaction_history, start_idx=None, end_idx=None,
                       price_col='Close', title='Trading Actions', 
                       figsize=(14, 7), save_path=None):
    """
    Plot price with buy/sell actions
    
    Args:
        df (pandas.DataFrame): Price data with DateTimeIndex
        transaction_history (pandas.DataFrame): Transaction history from the agent
        start_idx (int): Starting index for plotting
        end_idx (int): Ending index for plotting
        price_col (str): Column to use for price data
        title (str): Title for the plot
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Slice dataframe if indices provided
    if start_idx is not None or end_idx is not None:
        df = df.iloc[slice(start_idx, end_idx)]
    
    # Plot price
    plt.plot(df.index, df[price_col], label=price_col, color='blue', alpha=0.6)
    
    # Extract buy and sell points
    buy_indices = transaction_history[transaction_history['type'] == 'buy']['step']
    sell_indices = transaction_history[transaction_history['type'] == 'sell']['step']
    
    # Plot buy and sell points
    buy_dates = [df.index[i] for i in buy_indices if i < len(df)]
    buy_prices = [df[price_col].iloc[i] for i in buy_indices if i < len(df)]
    
    sell_dates = [df.index[i] for i in sell_indices if i < len(df)]
    sell_prices = [df[price_col].iloc[i] for i in sell_indices if i < len(df)]
    
    plt.scatter(buy_dates, buy_prices, color='green', label='Buy', marker='^', s=100)
    plt.scatter(sell_dates, sell_prices, color='red', label='Sell', marker='v', s=100)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_portfolio_performance(transaction_history, price_data, 
                             initial_balance=10000, commission=0.001,
                             title='Portfolio Performance', figsize=(14, 10),
                             save_path=None):
    """
    Plot portfolio performance over time
    
    Args:
        transaction_history (pandas.DataFrame): Transaction history from the agent
        price_data (pandas.DataFrame): Price data with DateTimeIndex
        initial_balance (float): Initial balance
        commission (float): Trading commission
        title (str): Title for the plot
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
    """
    # Create figure and subplots
    fig, axs = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Get price data
    prices = price_data['Close']
    
    # Calculate portfolio value at each step
    balance = initial_balance
    crypto_held = 0
    portfolio_value = [initial_balance]
    portfolio_dates = [price_data.index[0]]
    
    # Buy and hold strategy for comparison
    buy_hold_units = initial_balance / prices.iloc[0]
    buy_hold_value = [initial_balance]
    
    for i in range(1, len(prices)):
        # Update buy and hold value
        buy_hold_value.append(buy_hold_units * prices.iloc[i])
        
        # Check if there was a transaction at this step
        transactions = transaction_history[transaction_history['step'] == i]
        
        if not transactions.empty:
            for _, tx in transactions.iterrows():
                if tx['type'] == 'buy':
                    # Buy transaction
                    crypto_held = tx['crypto_held']
                    balance = tx['balance']
                elif tx['type'] == 'sell':
                    # Sell transaction
                    crypto_held = tx['crypto_held']
                    balance = tx['balance']
        
        # Calculate portfolio value
        current_value = balance + crypto_held * prices.iloc[i]
        portfolio_value.append(current_value)
        portfolio_dates.append(price_data.index[i])
    
    # Convert to numpy arrays for calculations
    portfolio_value = np.array(portfolio_value)
    buy_hold_value = np.array(buy_hold_value)
    
    # Plot portfolio value
    axs[0].plot(portfolio_dates, portfolio_value, label='Portfolio Value', color='blue')
    axs[0].plot(portfolio_dates, buy_hold_value, label='Buy & Hold', color='green', linestyle='--')
    axs[0].set_title(title)
    axs[0].set_ylabel('Portfolio Value ($)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot returns compared to buy & hold
    returns = (portfolio_value / initial_balance - 1) * 100
    buy_hold_returns = (buy_hold_value / initial_balance - 1) * 100
    
    axs[1].plot(portfolio_dates, returns, label='Strategy Returns (%)', color='blue')
    axs[1].plot(portfolio_dates, buy_hold_returns, label='Buy & Hold Returns (%)', color='green', linestyle='--')
    axs[1].set_ylabel('Returns (%)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot relative performance (ratio of portfolio value to buy & hold)
    relative_perf = (portfolio_value / buy_hold_value - 1) * 100
    
    axs[2].plot(portfolio_dates, relative_perf, label='Relative Performance (%)', color='purple')
    axs[2].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axs[2].set_ylabel('Relative to Buy & Hold (%)')
    axs[2].set_xlabel('Date')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Calculate and return performance metrics
    final_return = returns[-1]
    buy_hold_return = buy_hold_returns[-1]
    outperformance = final_return - buy_hold_return
    max_drawdown = np.min(returns) if np.min(returns) < 0 else 0
    
    metrics = {
        'final_portfolio_value': portfolio_value[-1],
        'final_return_pct': final_return,
        'buy_hold_return_pct': buy_hold_return,
        'outperformance_pct': outperformance,
        'max_drawdown_pct': max_drawdown
    }
    
    return metrics


def create_trading_animation(df, transaction_history, price_col='Close', window_size=50,
                           title='Cryptocurrency Trading Animation', figsize=(12, 6),
                           save_path=None, fps=10, dpi=100):
    """
    Create an animation of the trading process
    
    Args:
        df (pandas.DataFrame): Price data with DateTimeIndex
        transaction_history (pandas.DataFrame): Transaction history from the agent
        price_col (str): Column to use for price data
        window_size (int): Number of data points to show in the moving window
        title (str): Title for the animation
        figsize (tuple): Figure size
        save_path (str): Path to save the animation
        fps (int): Frames per second
        dpi (int): DPI for the animation
    
    Returns:
        matplotlib.animation.FuncAnimation: Animation object
    """
    # Extract price data
    prices = df[price_col].values
    dates = df.index
    
    # Create transaction dictionaries for faster lookup
    buy_dict = {step: price for step, price in zip(
        transaction_history[transaction_history['type'] == 'buy']['step'],
        transaction_history[transaction_history['type'] == 'buy']['price']
    )}
    
    sell_dict = {step: price for step, price in zip(
        transaction_history[transaction_history['type'] == 'sell']['step'],
        transaction_history[transaction_history['type'] == 'sell']['price']
    )}
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Function to update the plot for each frame
    def update(frame):
        ax.clear()
        
        # Get data for the current window
        start = max(0, frame - window_size)
        end = frame + 1
        
        # Calculate visible data range
        visible_prices = prices[start:end]
        visible_dates = dates[start:end]
        
        # Plot price data
        ax.plot(visible_dates, visible_prices, color='blue', label=price_col)
        
        # Add buy/sell markers for visible range
        for i in range(start, end):
            if i in buy_dict:
                ax.scatter(dates[i], prices[i], color='green', marker='^', s=100)
            elif i in sell_dict:
                ax.scatter(dates[i], prices[i], color='red', marker='v', s=100)
        
        # Add labels and title
        current_date = dates[frame].strftime('%Y-%m-%d %H:%M') if hasattr(dates[frame], 'strftime') else str(dates[frame])
        ax.set_title(f"{title}\nDate: {current_date}, Price: ${prices[frame]:.2f}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        
        # Add a rectangle to highlight the current price
        rect = Rectangle((dates[frame], prices[frame] - prices[frame]*0.01), 
                        width=pd.Timedelta(days=1), 
                        height=prices[frame]*0.02,
                        color='yellow', alpha=0.5)
        ax.add_patch(rect)
        
        # Set auto-scale for y-axis with some margin
        price_range = max(visible_prices) - min(visible_prices)
        y_min = min(visible_prices) - price_range * 0.1
        y_max = max(visible_prices) + price_range * 0.1
        ax.set_ylim(y_min, y_max)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Add legend for buy/sell actions
        buy_patch = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='Buy')
        sell_patch = plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, label='Sell')
        price_line = plt.Line2D([0], [0], color='blue', lw=2, label=price_col)
        
        ax.legend(handles=[price_line, buy_patch, sell_patch], loc='upper left')
        
        plt.tight_layout()
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=range(len(prices)), interval=1000/fps)
    
    # Save animation if path provided
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        ani.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi)
        print(f"Animation saved to {save_path}")
    
    plt.close(fig)  # Close the figure to avoid displaying it twice
    
    return ani


def plot_performance_metrics(metrics, title='Trading Performance Metrics', 
                         figsize=(10, 6), save_path=None):
    """
    Plot performance metrics for the trading agent
    
    Args:
        metrics (dict): Dictionary of performance metrics
        title (str): Title for the plot
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Create bar chart for performance metrics
    metrics_to_plot = {
        'Return (%)': metrics.get('final_return_pct', 0),
        'Buy & Hold (%)': metrics.get('buy_hold_return_pct', 0),
        'Outperformance (%)': metrics.get('outperformance_pct', 0),
        'Max Drawdown (%)': metrics.get('max_drawdown_pct', 0)
    }
    
    # Plot
    bars = plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
    
    # Color bars based on positive/negative values
    for i, bar in enumerate(bars):
        value = list(metrics_to_plot.values())[i]
        if value < 0:
            bar.set_color('red')
        else:
            bar.set_color('green')
    
    # Add value labels on top of bars
    for i, (bar, value) in enumerate(zip(bars, metrics_to_plot.values())):
        plt.text(i, value + (1 if value >= 0 else -1), 
                f'{value:.2f}%', ha='center', va='bottom' if value >= 0 else 'top')
    
    plt.title(title)
    plt.ylabel('Percentage (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show() 