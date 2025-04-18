import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class CryptoTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000.0, commission=0.001):
        super(CryptoTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.commission = commission
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(self.df.columns),),
            dtype=np.float32
        )
        
        # Reset environment
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.total_profit = 0.0
        self.trades = []
        
        return self._get_observation()
    
    def step(self, action):
        # Get current price
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Convert action to trading amount (-1 to 1)
        trade_amount = float(action[0])
        
        # Calculate amount to buy/sell
        if trade_amount > 0:  # Buy
            amount = trade_amount * self.balance / current_price
            cost = amount * current_price * (1 + self.commission)
            
            if cost > self.balance:
                amount = self.balance / (current_price * (1 + self.commission))
                cost = self.balance
            
            self.balance -= cost
            self.crypto_held += amount
            
            self.trades.append({
                'step': self.current_step,
                'type': 'buy',
                'amount': amount,
                'price': current_price,
                'cost': cost
            })
        
        elif trade_amount < 0:  # Sell
            amount = -trade_amount * self.crypto_held
            
            if amount > self.crypto_held:
                amount = self.crypto_held
            
            revenue = amount * current_price * (1 - self.commission)
            
            self.balance += revenue
            self.crypto_held -= amount
            
            self.trades.append({
                'step': self.current_step,
                'type': 'sell',
                'amount': amount,
                'price': current_price,
                'revenue': revenue
            })
        
        # Calculate reward
        portfolio_value = self.balance + self.crypto_held * current_price
        reward = portfolio_value - self.initial_balance - self.total_profit
        self.total_profit += reward
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Get observation
        obs = self._get_observation()
        
        # Additional info
        info = {
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'portfolio_value': portfolio_value,
            'current_price': current_price,
            'total_profit': self.total_profit
        }
        
        return obs, reward, done, info
    
    def _get_observation(self):
        return self.df.iloc[self.current_step].values
    
    def get_transaction_history(self):
        return pd.DataFrame(self.trades) 