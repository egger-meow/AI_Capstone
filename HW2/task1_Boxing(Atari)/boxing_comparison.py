import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation
import cv2
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Create necessary directories
os.makedirs("comparison_videos", exist_ok=True)

def create_env(env_id="ALE/Boxing-v5"):
    """Create the Boxing environment with proper wrappers"""
    env = gym.make(env_id, render_mode="rgb_array")
    env = AtariWrapper(env)
    return env

def load_models(model_paths):
    """Load models from saved files"""
    models = {}
    for name, path in model_paths.items():
        if "DQN" in name:
            models[name] = DQN.load(path)
        elif "A2C" in name:
            models[name] = A2C.load(path)
        elif "PPO" in name:
            models[name] = PPO.load(path)
    return models

def create_side_by_side_comparison(models, env_id="ALE/Boxing-v5", num_episodes=3, 
                                  seed=42, output_path="comparison_videos/side_by_side.mp4"):
    """Create a video comparing multiple models side by side on the same scenarios"""
    # Create environments (one for each model)
    envs = {name: create_env(env_id) for name in models.keys()}
    
    # Setup for recording
    fps = 30
    num_models = len(models)
    
    # Get frame dimensions
    test_env = list(envs.values())[0]
    test_obs, _ = test_env.reset(seed=seed)
    test_frame = test_env.render()
    frame_height, frame_width, _ = test_frame.shape
    
    # Create video writer for combined visualization
    # Calculate dimensions for the grid layout
    if num_models <= 2:
        grid_cols, grid_rows = num_models, 1
    else:
        grid_cols, grid_rows = 2, (num_models + 1) // 2
    
    # Calculate dimensions for the final video
    margin = 10  # Pixels between frames
    total_width = grid_cols * frame_width + (grid_cols - 1) * margin
    total_height = grid_rows * frame_height + (grid_rows - 1) * margin
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (total_width, total_height))
    
    for episode in range(num_episodes):
        print(f"Recording episode {episode+1}/{num_episodes}")
        
        # Reset all environments with the same seed for fair comparison
        current_seed = seed + episode
        observations = {name: env.reset(seed=current_seed)[0] for name, env in envs.items()}
        
        # Track episode data
        frames = {name: [] for name in models.keys()}
        rewards = {name: 0 for name in models.keys()}
        dones = {name: False for name in models.keys()}
        
        # Run the episode for all agents
        max_steps = 10000  # Safety limit
        step = 0
        
        all_done = False
        while not all_done and step < max_steps:
            # Create the combined frame
            combined_frame = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
            
            # Get actions and step environments
            for i, (name, model) in enumerate(models.items()):
                if not dones[name]:
                    # Get the next action
                    action, _ = model.predict(observations[name], deterministic=True)
                    next_obs, reward, terminated, truncated, _ = envs[name].step(action)
                    
                    # Update tracking
                    observations[name] = next_obs
                    rewards[name] += reward
                    dones[name] = terminated or truncated
                
                # Render the current state
                frame = envs[name].render()
                
                # Add text overlay with model name and current score
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(frame, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Score: {rewards[name]}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Store the frame
                frames[name].append(frame)
                
                # Calculate position in grid
                grid_col = i % grid_cols
                grid_row = i // grid_cols
                
                # Calculate position in the combined frame
                x_offset = grid_col * (frame_width + margin)
                y_offset = grid_row * (frame_height + margin)
                
                # Insert into combined frame
                combined_frame[y_offset:y_offset+frame_height, x_offset:x_offset+frame_width] = frame
            
            # Write combined frame to video
            video_writer.write(combined_frame)
            
            # Check if all environments are done
            all_done = all(dones.values())
            step += 1
        
        # Reset environments for next episode
        for env in envs.values():
            env.close()
        envs = {name: create_env(env_id) for name in models.keys()}
        
        print(f"Episode {episode+1} completed. Final scores:")
        for name, reward in rewards.items():
            print(f"  {name}: {reward}")
    
    # Clean up
    video_writer.release()
    for env in envs.values():
        env.close()
    
    print(f"Comparison video saved to {output_path}")
    return output_path

def create_performance_metrics_table(models, env_id="ALE/Boxing-v5", num_eval_episodes=10):
    """Evaluate models and create a performance comparison table"""
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        env = create_env(env_id)
        
        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []
        knockouts = 0  # Count number of knockouts (reward of 100)
        
        for episode in range(num_eval_episodes):
            obs, _ = env.reset(seed=episode)
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if episode_reward >= 100:
                knockouts += 1
        
        # Calculate statistics
        results[name] = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "knockouts": knockouts,
            "knockout_ratio": knockouts / num_eval_episodes
        }
        
        env.close()
    
    # Create a nicely formatted table
    column_headers = ["Algorithm", "Mean Reward", "Min/Max", "Knockouts", "Avg. Length"]
    table_data = []
    
    for name, metrics in results.items():
        table_data.append([
            name,
            f"{metrics['mean_reward']:.1f} ± {metrics['std_reward']:.1f}",
            f"{metrics['min_reward']:.0f} / {metrics['max_reward']:.0f}",
            f"{metrics['knockouts']}/{num_eval_episodes} ({metrics['knockout_ratio']*100:.0f}%)",
            f"{metrics['mean_length']:.1f}"
        ])
    
    # Print table
    print("\n===== Model Performance Comparison =====")
    # Print header
    header = "| " + " | ".join(column_headers) + " |"
    separator = "|" + "|".join(["-" * (len(h) + 2) for h in column_headers]) + "|"
    print(header)
    print(separator)
    
    # Print data rows
    for row in table_data:
        print("| " + " | ".join(row) + " |")
    
    # Create a visual bar chart
    plt.figure(figsize=(12, 6))
    names = list(results.keys())
    means = [results[name]["mean_reward"] for name in names]
    stds = [results[name]["std_reward"] for name in names]
    knockouts = [results[name]["knockout_ratio"] * 100 for name in names]
    
    # Plot reward bars
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(names, means, yerr=stds, capsize=10, color="skyblue")
    ax1.set_title("Mean Episode Reward", fontsize=14)
    ax1.set_ylabel("Reward", fontsize=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Plot knockout percentage
    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(names, knockouts, color="salmon")
    ax2.set_title("Knockout Percentage", fontsize=14)
    ax2.set_ylabel("Percentage (%)", fontsize=12)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig("comparison_videos/model_performance_comparison.png", dpi=300)
    plt.close()
    
    return results

def compare_models_from_checkpoints():
    """Load models from checkpoints and create comparative visualizations"""
    # Define paths to model checkpoints
    algorithm_names = ["DQN", "A2C", "PPO"]
    model_checkpoints = {}
    
    # Try to find the model files in both possible locations
    for algo in algorithm_names:
        # Look in models directory first
        models_path = f"./models/{algo}/best_model.zip"
        final_model_path = f"./models/{algo}/final_model.zip"
        checkpoint_path = f"./checkpoints/{algo}/final_model.zip"
        
        # Check all possible locations
        if os.path.exists(models_path):
            model_checkpoints[algo] = models_path
        elif os.path.exists(final_model_path):
            model_checkpoints[algo] = final_model_path
        elif os.path.exists(checkpoint_path):
            model_checkpoints[algo] = checkpoint_path
        else:
            # Look for any .zip files in both directories
            for base_dir in [f"./models/{algo}/", f"./checkpoints/{algo}/"]:
                if os.path.exists(base_dir):
                    checkpoint_files = [f for f in os.listdir(base_dir) if f.endswith(".zip")]
                    if checkpoint_files:
                        # Sort by timestep number if available
                        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]) if "_" in x and x.split("_")[-1].split(".")[0].isdigit() else 0, reverse=True)
                        model_checkpoints[algo] = os.path.join(base_dir, checkpoint_files[0])
                        break
    
    if not model_checkpoints:
        print("No model checkpoints found in either ./models/ or ./checkpoints/ directories! Please train models first.")
        return None
    
    print(f"\nFound {len(model_checkpoints)} model checkpoints:")
    for name, path in model_checkpoints.items():
        print(f"  {name}: {path}")
    
    # Load models
    models = load_models(model_checkpoints)
    
    # Create side-by-side comparison video
    print("\nCreating side-by-side comparison video...")
    comparison_path = create_side_by_side_comparison(
        models, 
        num_episodes=3, 
        output_path="comparison_videos/algorithm_comparison.mp4"
    )
    
    # Create performance metrics table
    print("\nEvaluating model performance...")
    performance_results = create_performance_metrics_table(models, num_eval_episodes=10)
    
    print("\nComparison completed! Results and videos are available in the comparison_videos/ directory.")
    return models, performance_results

if __name__ == "__main__":
    # Run the comparison
    models, results = compare_models_from_checkpoints() 