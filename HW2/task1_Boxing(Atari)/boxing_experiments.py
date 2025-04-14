import os
import numpy as np
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

# Create log directory
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("videos/", exist_ok=True)
os.makedirs("models/", exist_ok=True)

# Custom callback for plotting and saving progress
class ProgressCallback(BaseCallback):
    def __init__(self, check_freq=1000, save_path="./models/", verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        
    def _init_callback(self):
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Get the monitor's data
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            if len(x) > 0:
                # Mean reward over last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")

                # New best model, save the agent
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(f"{self.save_path}{self.model.__class__.__name__}_{self.num_timesteps}")
            
            return True
        return True

def make_env(env_id, rank, seed=0, monitor_dir=None):
    """
    Create a wrapped, monitored gym.Env for Atari
    """
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = AtariWrapper(env)
        if monitor_dir:
            env = Monitor(env, os.path.join(monitor_dir, str(rank)), allow_early_resets=True)
        env.reset(seed=seed + rank)
        return env
    return _init

def create_vec_env(env_id, num_envs=1):
    """Create a vectorized environment for training"""
    env = DummyVecEnv([make_env(env_id, i, monitor_dir=log_dir) for i in range(num_envs)])
    return env

def record_video(model, env_id, video_length=1000, prefix="", video_folder="videos/"):
    """
    Record a video of an agent's performance
    """
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
    eval_env = VecVideoRecorder(
        eval_env, 
        video_folder=video_folder,
        record_video_trigger=lambda x: x == 0, 
        video_length=video_length,
        name_prefix=f"{prefix}_{model.__class__.__name__}"
    )

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, dones, _ = eval_env.step(action)
        if dones.any():
            obs = eval_env.reset()
    
    eval_env.close()

def collect_frames(model, env_id, num_episodes=1):
    """Collect frames from the environment for visualization"""
    env = gym.make(env_id, render_mode="rgb_array")
    env = AtariWrapper(env)
    
    all_episode_frames = []
    all_episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)  # Different seed for variety
        episode_frames = []
        episode_reward = 0
        
        done = False
        while not done:
            # Capture raw frame from env before any preprocessing
            frame = env.render()
            episode_frames.append(frame)
            
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        all_episode_frames.append(episode_frames)
        all_episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{num_episodes} - Score: {episode_reward}")
    
    env.close()
    return all_episode_frames, all_episode_rewards

def create_episode_animation(frames, episode_reward, filename="episode.mp4"):
    """Create animation from collected frames"""
    fig = plt.figure(figsize=(10, 8), dpi=150)
    plt.axis('off')
    plt.title(f"Total Reward: {episode_reward}", fontsize=16)
    im = plt.imshow(frames[0], interpolation="bilinear")
    
    def update_frame(i):
        im.set_array(frames[i])
        return [im]
    
    ani = animation.FuncAnimation(fig, update_frame, frames=len(frames), interval=50, blit=True)
    ani.save(filename, writer="ffmpeg", fps=30, dpi=150)
    plt.close()
    return filename

def train_and_evaluate(algorithm, env_id, total_timesteps=500000, eval_freq=10000):
    """Train a model and evaluate it periodically"""
    # Create training environment
    env = create_vec_env(env_id)
    
    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env=create_vec_env(env_id),
        n_eval_episodes=5,
        eval_freq=eval_freq,
        log_path=log_dir,
        best_model_save_path=f"./models/{algorithm.__name__}/",
        deterministic=True
    )
    
    # Create progress callback
    progress_callback = ProgressCallback(check_freq=eval_freq, save_path=f"./models/{algorithm.__name__}/")
    
    # Initialize model
    if algorithm == DQN:
        model = algorithm("CnnPolicy", env, verbose=1, 
                         buffer_size=100000, 
                         learning_rate=1e-4,
                         exploration_fraction=0.1,
                         tensorboard_log=f"./logs/{algorithm.__name__}_tensorboard/")
    elif algorithm == A2C:
        model = algorithm("CnnPolicy", env, verbose=1, 
                        learning_rate=7e-4,
                        n_steps=5,
                        tensorboard_log=f"./logs/{algorithm.__name__}_tensorboard/")
    elif algorithm == PPO:
        model = algorithm("CnnPolicy", env, verbose=1, 
                        learning_rate=2.5e-4,
                        n_steps=128,
                        tensorboard_log=f"./logs/{algorithm.__name__}_tensorboard/")
    
    # Train the model with callbacks
    print(f"\n======= Training {algorithm.__name__} for {total_timesteps} timesteps =======\n")
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[eval_callback, progress_callback],
        progress_bar=True
    )
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds\n")
    
    # Save the final model
    model.save(f"./models/{algorithm.__name__}/final_model")
    
    # Record video of trained agent
    record_video(model, env_id, prefix="final")
    
    # Create frame animations
    print("Collecting frames for visualization...")
    episode_frames, episode_rewards = collect_frames(model, env_id, num_episodes=3)
    
    for i, (frames, reward) in enumerate(zip(episode_frames, episode_rewards)):
        filename = f"videos/{algorithm.__name__}_episode_{i+1}_reward_{reward}.mp4"
        create_episode_animation(frames, reward, filename)
        print(f"Created animation: {filename}")
    
    # Final evaluation
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"\n===== Final {algorithm.__name__} Evaluation =====")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()
    return model, mean_reward, std_reward

def compare_algorithms(env_id="ALE/Boxing-v5", timesteps=500000):
    """Run training and evaluation for multiple algorithms and compare results"""
    results = {}
    
    # Define the algorithms to test
    algorithms = [DQN, A2C, PPO]
    
    for algorithm in algorithms:
        print(f"\n\n{'='*20} TRAINING {algorithm.__name__} {'='*20}\n")
        model, mean_reward, std_reward = train_and_evaluate(algorithm, env_id, timesteps)
        results[algorithm.__name__] = {
            "mean_reward": mean_reward,
            "std_reward": std_reward
        }
    
    # Plot comparative results
    plt.figure(figsize=(12, 8))
    algorithms = list(results.keys())
    means = [results[alg]["mean_reward"] for alg in algorithms]
    stds = [results[alg]["std_reward"] for alg in algorithms]
    
    plt.bar(algorithms, means, yerr=stds, capsize=10)
    plt.title("Algorithm Performance Comparison", fontsize=16)
    plt.ylabel("Mean Episode Reward", fontsize=14)
    plt.xlabel("Algorithm", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + 0.1, f"{mean:.2f} ± {std:.2f}", 
                ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("algorithm_comparison.png", dpi=300)
    plt.close()
    
    # Print final results summary
    print("\n\n===== FINAL ALGORITHM COMPARISON =====")
    for algorithm, result in results.items():
        print(f"{algorithm}: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
    
    return results

if __name__ == "__main__":
    # Set the total timesteps for training
    TOTAL_TIMESTEPS = 100000  # Adjust as needed
    
    # Compare different algorithms
    results = compare_algorithms(timesteps=TOTAL_TIMESTEPS)
    
    print("\nExperiments completed! Check the videos/ directory for visualizations.") 