import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from enum import Enum


class Actions(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


class CryptoTradingEnv(gym.Env):
    """
    A simple cryptocurrency trading environment for OpenAI gym
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=20, render_mode=None, initial_balance=10000, 
                 commission=0.001, reward_scaling=1.0):
        """
        Args:
            df (pandas.DataFrame): DataFrame with cryptocurrency price data
            window_size (int): Number of previous price points to include in state
            render_mode (str): Mode for rendering the environment
            initial_balance (float): Initial amount of money for trading
            commission (float): Trading commission percentage
            reward_scaling (float): Scaling factor for rewards
        """
        super(CryptoTradingEnv, self).__init__()
        
        self.df = df
        self.window_size = window_size
        self.render_mode = render_mode
        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_scaling = reward_scaling
        
        # Action space: [HOLD, BUY, SELL]
        self.action_space = spaces.Discrete(len(Actions))
        
        # Observation space: [prices, technical indicators, balance, holdings]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size + 2,), dtype=np.float32
        )
        
        # Episode variables
        self.current_step = 0
        self.balance = initial_balance
        self.crypto_held = 0
        self.net_worth = initial_balance
        self.transaction_history = []

    def _next_observation(self):
        """
        Get the next state observation
        """
        # Get the window of cryptocurrency prices
        end = self.current_step + 1
        start = end - self.window_size
        
        # Handle the case for start < 0 (beginning of data)
        if start < 0:
            zeros = np.zeros((-start, 1))
            price_window = np.concatenate((zeros, self.df['Close'].values[0:end, np.newaxis]))
        else:
            price_window = self.df['Close'].values[start:end]
        
        # Normalize the price data
        normalized_price = price_window / price_window[-1] - 1
        
        # Add balance and holdings to the observation
        observation = np.append(
            normalized_price,
            [
                self.balance / self.initial_balance - 1, 
                self.crypto_held * self.df['Close'].values[self.current_step] / self.initial_balance
            ]
        )
        
        return observation.astype(np.float32)

    def _calculate_reward(self, action):
        """
        Calculate the reward for the current step
        """
        # Current price
        current_price = self.df['Close'].values[self.current_step]
        
        # Previous net worth
        prev_net_worth = self.net_worth
        
        # Calculate current net worth
        self.net_worth = self.balance + self.crypto_held * current_price
        
        # Calculate profit/loss for this step
        profit = self.net_worth - prev_net_worth
        
        # Reward is profit scaled by reward_scaling
        reward = profit * self.reward_scaling
        
        return reward
    
    def _take_action(self, action):
        """
        Execute the action in the environment
        """
        # Current price
        current_price = self.df['Close'].values[self.current_step]
        
        # Execute action
        if action == Actions.BUY.value and self.balance > 0:
            # Calculate maximum amount of crypto to buy
            max_crypto_to_buy = self.balance / current_price
            
            # Apply commission
            crypto_bought = max_crypto_to_buy * (1 - self.commission)
            
            # Update balance and holdings
            self.balance = 0
            self.crypto_held += crypto_bought
            
            # Record transaction
            self.transaction_history.append({
                'step': self.current_step,
                'type': 'buy',
                'price': current_price,
                'amount': crypto_bought,
                'balance': self.balance,
                'crypto_held': self.crypto_held
            })
            
        elif action == Actions.SELL.value and self.crypto_held > 0:
            # Calculate selling amount
            crypto_sold = self.crypto_held
            
            # Apply commission
            self.balance += crypto_sold * current_price * (1 - self.commission)
            self.crypto_held = 0
            
            # Record transaction
            self.transaction_history.append({
                'step': self.current_step,
                'type': 'sell',
                'price': current_price,
                'amount': crypto_sold,
                'balance': self.balance,
                'crypto_held': self.crypto_held
            })
            
    def step(self, action):
        """
        Take a step in the environment
        """
        # Execute action
        self._take_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Check if done (end of data)
        done = self.current_step >= len(self.df) - 1
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Get next observation
        obs = self._next_observation()
        
        # Info dictionary for metrics
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'current_price': self.df['Close'].values[self.current_step]
        }
        
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode
        """
        super().reset(seed=seed)
        
        # Reset variables
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.net_worth = self.initial_balance
        self.current_step = self.window_size
        self.transaction_history = []
        
        return self._next_observation(), {}

    def render(self, mode='human'):
        """
        Render the environment
        """
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Price: ${self.df['Close'].values[self.current_step]:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Crypto Held: {self.crypto_held:.6f}")
            print(f"Net Worth: ${self.net_worth:.2f}")
            print("------------------------")
            
    def get_transaction_history(self):
        """
        Return the transaction history for analysis
        """
        return pd.DataFrame(self.transaction_history) 