import os
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics during training
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.training_env = None
        
    def _on_step(self):
        # Log additional info
        if self.training_env is not None:
            info = self.training_env.get_attr('info')[0]
            
            # Log net worth
            if 'net_worth' in info:
                self.logger.record('trading/net_worth', info['net_worth'])
            
            # Log balance and holdings
            if 'balance' in info and 'crypto_held' in info:
                self.logger.record('trading/balance', info['balance'])
                self.logger.record('trading/crypto_held', info['crypto_held'])
                
            # Log current price
            if 'current_price' in info:
                self.logger.record('trading/current_price', info['current_price'])
                
        return True


def create_agent(env, tensorboard_log="./logs/", learning_rate=0.0001, buffer_size=10000, 
                learning_starts=1000, batch_size=64, tau=1.0, gamma=0.99, train_freq=4, 
                gradient_steps=1, target_update_interval=10000, exploration_fraction=0.1,
                exploration_initial_eps=1.0, exploration_final_eps=0.05, max_grad_norm=10,
                policy_kwargs=None):
    """
    Create and configure a DQN agent for the crypto trading environment
    
    Args:
        env: The gym environment
        tensorboard_log: Directory for tensorboard logs
        learning_rate: Learning rate
        buffer_size: Size of the replay buffer
        learning_starts: How many steps before learning starts
        batch_size: Batch size for training
        tau: Soft update coefficient for target network update
        gamma: Discount factor
        train_freq: Update the model every train_freq steps
        gradient_steps: How many gradient steps to do after each rollout
        target_update_interval: Update the target network every interval
        exploration_fraction: Fraction of exploration
        exploration_initial_eps: Initial value of random action probability
        exploration_final_eps: Final value of random action probability
        max_grad_norm: Maximum value for gradient clipping
        policy_kwargs: Arguments to be passed to the policy on creation
    
    Returns:
        A configured DQN agent
    """
    # Create default policy kwargs if none provided
    if policy_kwargs is None:
        policy_kwargs = {
            "net_arch": [64, 64],
            "activation_fn": th.nn.ReLU
        }
    
    # Create the DQN agent
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        max_grad_norm=max_grad_norm,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    return model


def train_agent(model, timesteps, callbacks=None, progress_bar=True):
    """
    Train the agent
    
    Args:
        model: The DQN model to train
        timesteps: Number of timesteps to train for
        callbacks: List of callbacks for training
        progress_bar: Whether to display a progress bar during training
    
    Returns:
        The trained model
    """
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=progress_bar)
    return model


def evaluate_agent(model, env, num_episodes=5):
    """
    Evaluate a trained agent
    
    Args:
        model: The trained model
        env: The evaluation environment
        num_episodes: Number of episodes to evaluate
    
    Returns:
        A list of episode rewards and a list of episode infos
    """
    episode_rewards = []
    episode_infos = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        info_history = []
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            info_history.append(info)
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_infos.append(info_history)
    
    return episode_rewards, episode_infos


def save_agent(model, filepath="models/dqn_crypto_trading"):
    """
    Save the trained agent
    
    Args:
        model: The model to save
        filepath: Path to save the model to
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the model
    model.save(filepath)


def load_agent(filepath="models/dqn_crypto_trading", env=None):
    """
    Load a trained agent
    
    Args:
        filepath: Path to the saved model
        env: Environment to use with the loaded model
    
    Returns:
        The loaded model
    """
    return DQN.load(filepath, env=env) 