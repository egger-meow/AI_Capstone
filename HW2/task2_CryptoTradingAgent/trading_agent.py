import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm


class TradingCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, verbose=1):
        super(TradingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Save the model
            model_path = os.path.join(self.log_dir, f"model_{self.n_calls}")
            self.model.save(model_path)
            
            # Evaluate the model
            mean_reward, _ = evaluate_policy(self.model, self.training_env, n_eval_episodes=5)
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.log_dir, "best_model"))
            
            if self.verbose > 0:
                print(f"Step: {self.n_calls}")
                print(f"Mean reward: {mean_reward:.2f}")
                print(f"Best mean reward: {self.best_mean_reward:.2f}")
        
        return True


class TradingAgent:
    def __init__(self, env, algorithm="sac", **kwargs):
        """
        Initialize the trading agent
        
        Args:
            env: Trading environment
            algorithm: RL algorithm to use ("sac", "td3", or "ddpg")
            **kwargs: Additional arguments for the algorithm
        """
        self.env = env
        self.algorithm = algorithm.lower()
        
        # Set up the algorithm
        if self.algorithm == "sac":
            self.model = SAC("MlpPolicy", env, verbose=1, **kwargs)
        elif self.algorithm == "td3":
            self.model = TD3("MlpPolicy", env, verbose=1, **kwargs)
        elif self.algorithm == "ddpg":
            self.model = DDPG("MlpPolicy", env, verbose=1, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def train(self, total_timesteps, callback=None):
        """
        Train the agent
        
        Args:
            total_timesteps: Number of timesteps to train for
            callback: Callback function for logging and saving
        """
        # Create progress bar
        progress_bar = tqdm(total=total_timesteps, desc="Training Progress")
        
        # Custom callback to update progress bar
        class ProgressCallback(BaseCallback):
            def __init__(self, progress_bar):
                super(ProgressCallback, self).__init__()
                self.progress_bar = progress_bar
                
            def _on_step(self) -> bool:
                self.progress_bar.update(1)
                return True
        
        # Combine callbacks
        if callback is not None:
            callbacks = [callback, ProgressCallback(progress_bar)]
        else:
            callbacks = [ProgressCallback(progress_bar)]
            
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10
        )
        
        # Close progress bar
        progress_bar.close()

    def save(self, path):
        """
        Save the model
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)

    def load(self, path):
        """
        Load a saved model
        
        Args:
            path: Path to the saved model
        """
        if self.algorithm == "sac":
            self.model = SAC.load(path)
        elif self.algorithm == "td3":
            self.model = TD3.load(path)
        elif self.algorithm == "ddpg":
            self.model = DDPG.load(path)

    def predict(self, observation, deterministic=True):
        """
        Predict an action given an observation
        
        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic actions
            
        Returns:
            action: Predicted action
        """
        return self.model.predict(observation, deterministic=deterministic)

    def evaluate(self, env, n_episodes=10):
        """
        Evaluate the agent
        
        Args:
            env: Environment to evaluate in
            n_episodes: Number of episodes to evaluate
            
        Returns:
            mean_reward: Mean reward over episodes
            std_reward: Standard deviation of rewards
        """
        episode_rewards = []
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.predict(obs)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards) 