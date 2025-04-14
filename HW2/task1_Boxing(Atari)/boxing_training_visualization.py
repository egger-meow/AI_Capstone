import os
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import time

# Create directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("training_videos", exist_ok=True)
os.makedirs("training_stats", exist_ok=True)

class TrainingVisualizationCallback(BaseCallback):
    """
    Custom callback for capturing agent's performance at different training stages
    """
    def __init__(self, 
                 eval_env,
                 check_freq=50000, 
                 save_path="./checkpoints/", 
                 video_path="./training_videos/",
                 total_timesteps=500000,
                 verbose=1):
        super(TrainingVisualizationCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.video_path = video_path
        self.eval_env = eval_env
        self.total_timesteps = total_timesteps
        self.checkpoints = []
        self.rewards_history = []
        self.timesteps_history = []
        
    def _init_callback(self):
        # Create folders if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        if self.video_path is not None:
            os.makedirs(self.video_path, exist_ok=True)
            
    def _on_step(self):
        # Check if it's time to save a checkpoint
        if self.n_calls % self.check_freq == 0:
            checkpoint_path = f"{self.save_path}model_{self.num_timesteps}.zip"
            self.model.save(checkpoint_path)
            self.checkpoints.append((self.num_timesteps, checkpoint_path))
            
            # Run evaluation episodes
            rewards = self._evaluate_agent()
            mean_reward = np.mean(rewards)
            self.rewards_history.append(mean_reward)
            self.timesteps_history.append(self.num_timesteps)
            
            # Log progress
            progress_pct = (self.num_timesteps / self.total_timesteps) * 100
            print(f"Progress: {progress_pct:.1f}% | Timestep: {self.num_timesteps}/{self.total_timesteps}")
            print(f"Mean reward: {mean_reward:.2f}")
            
            # Save video of current performance
            self._save_video(self.num_timesteps)
            
            # Save learning curve so far
            self._plot_learning_curve()
        
        return True
    
    def _evaluate_agent(self, n_episodes=5):
        """Run n_episodes to evaluate current performance"""
        rewards = []
        
        for episode in range(n_episodes):
            episode_reward = 0
            obs, _ = self.eval_env.reset(seed=episode)
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            rewards.append(episode_reward)
        
        return rewards
    
    def _save_video(self, timestep, n_episodes=1):
        """Save video of agent performance at current checkpoint"""
        import cv2
        
        # Create video writer
        video_path = f"{self.video_path}boxing_training_{timestep}.mp4"
        fps = 30
        env = self.eval_env
        
        # Get the first observation to determine frame size
        obs, _ = env.reset(seed=0)
        frame = env.render()
        height, width, _ = frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        all_rewards = []
        
        # Play n episodes and record them
        for episode in range(n_episodes):
            obs, _ = env.reset(seed=episode)
            done = False
            episode_reward = 0
            
            while not done:
                frame = env.render()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Add progress info as text overlay
                progress_text = f"Timestep: {timestep} / {self.total_timesteps}"
                cv2.putText(frame, progress_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
                reward_text = f"Episode Reward: {episode_reward:.1f}"
                cv2.putText(frame, reward_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                video_writer.write(frame)
                
                # Get action and step environment
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            all_rewards.append(episode_reward)
        
        video_writer.release()
        return all_rewards
    
    def _plot_learning_curve(self):
        """Plot and save the learning curve with current data"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.timesteps_history, self.rewards_history, 'o-', linewidth=2)
        plt.title(f"Learning Progress - {self.model.__class__.__name__}", fontsize=16)
        plt.xlabel("Timesteps", fontsize=14)
        plt.ylabel("Mean Reward", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Mark the current point
        plt.scatter([self.timesteps_history[-1]], [self.rewards_history[-1]], 
                   color='red', s=100, zorder=5)
        
        plt.tight_layout()
        plt.savefig(f"training_stats/learning_curve_{self.model.__class__.__name__}.png", dpi=150)
        plt.close()

def create_env(env_id="ALE/Boxing-v5"):
    """Create the Boxing environment with proper wrappers"""
    env = gym.make(env_id, render_mode="rgb_array")
    env = AtariWrapper(env)
    return env

def visualize_training_process(algorithm=DQN, total_timesteps=500000, check_freq=50000, env_id="ALE/Boxing-v5"):
    """
    Train an agent and visualize its progress throughout training
    """
    # Create directories
    os.makedirs(f"checkpoints/{algorithm.__name__}", exist_ok=True)
    os.makedirs(f"training_videos/{algorithm.__name__}", exist_ok=True)
    
    # Create environments
    env = create_env(env_id)
    eval_env = create_env(env_id)
    
    # Initialize model with appropriate parameters
    if algorithm == DQN:
        model = algorithm(
            "CnnPolicy", 
            env, 
            verbose=1,
            buffer_size=100000,
            learning_rate=1e-4,
            exploration_fraction=0.1
        )
    elif algorithm == A2C:
        model = algorithm(
            "CnnPolicy", 
            env, 
            verbose=1,
            learning_rate=7e-4,
            n_steps=5
        )
    elif algorithm == PPO:
        model = algorithm(
            "CnnPolicy", 
            env, 
            verbose=1,
            learning_rate=2.5e-4,
            n_steps=128
        )
    
    # Create visualization callback
    vis_callback = TrainingVisualizationCallback(
        eval_env=eval_env,
        check_freq=check_freq,
        save_path=f"checkpoints/{algorithm.__name__}/",
        video_path=f"training_videos/{algorithm.__name__}/",
        total_timesteps=total_timesteps
    )
    
    # Train the model
    print(f"\n{'='*30} Starting {algorithm.__name__} Training {'='*30}")
    print(f"Total training steps: {total_timesteps}")
    print(f"Saving checkpoints every {check_freq} steps\n")
    
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=vis_callback, progress_bar=True)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save final model
    final_model_path = f"checkpoints/{algorithm.__name__}/final_model.zip"
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Create a final video
    print("Creating final performance video...")
    final_video_path = f"training_videos/{algorithm.__name__}/final_performance.mp4"
    
    # Generate learning curve evolution video
    create_learning_curve_video(
        vis_callback.timesteps_history,
        vis_callback.rewards_history,
        f"training_videos/{algorithm.__name__}/learning_curve_evolution.mp4",
        algorithm.__name__
    )
    
    # Clean up
    env.close()
    eval_env.close()
    
    return model, vis_callback.rewards_history, vis_callback.timesteps_history

def create_learning_curve_video(timesteps, rewards, output_path, algo_name):
    """Create a video showing the evolution of the learning curve"""
    import cv2
    import numpy as np
    
    # Setup plot
    fig = plt.figure(figsize=(10, 6), dpi=100)
    plt.title(f"Learning Progress - {algo_name}", fontsize=16)
    plt.xlabel("Timesteps", fontsize=14)
    plt.ylabel("Mean Reward", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Setup video writer
    width, height = fig.canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 5, (width, height))
    
    # For each checkpoint, show the growing learning curve
    for i in range(1, len(timesteps)+1):
        plt.clf()  # Clear the figure
        
        # Plot data up to current checkpoint
        plt.plot(timesteps[:i], rewards[:i], 'o-', linewidth=2, color='blue')
        plt.title(f"Learning Progress - {algo_name}", fontsize=16)
        plt.xlabel("Timesteps", fontsize=14)
        plt.ylabel("Mean Reward", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis limits to be consistent across frames
        plt.xlim(0, max(timesteps) * 1.1)
        plt.ylim(min(rewards) - 5, max(rewards) + 5)
        
        # Mark the current point
        if i > 0:
            plt.scatter([timesteps[i-1]], [rewards[i-1]], color='red', s=100, zorder=5)
            plt.text(timesteps[i-1], rewards[i-1] + 2, f"{rewards[i-1]:.1f}", 
                    horizontalalignment='center', fontsize=12)
        
        plt.tight_layout()
        
        # Convert plot to image
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Add to video
        video_writer.write(img)
    
    # Hold the final frame a bit longer
    for _ in range(20):
        video_writer.write(img)
    
    video_writer.release()
    plt.close()

if __name__ == "__main__":
    # Set parameters
    TOTAL_TIMESTEPS = 100000  # Total timesteps for training
    CHECK_FREQ = 10000       # How often to save checkpoints and videos
    
    # Train and visualize different algorithms
    algorithms = [DQN, A2C, PPO]
    
    for algorithm in algorithms:
        visualize_training_process(
            algorithm=algorithm,
            total_timesteps=TOTAL_TIMESTEPS,
            check_freq=CHECK_FREQ
        )
        
    print("\nAll training and visualization complete!") 