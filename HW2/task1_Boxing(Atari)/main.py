import os
import argparse
import sys
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Import our modules
from boxing_experiments import train_and_evaluate, compare_algorithms
from boxing_training_visualization import visualize_training_process
from boxing_comparison import compare_models_from_checkpoints

def print_header(text):
    """Print a nicely formatted header"""
    print("\n" + "=" * 80)
    print(f"{text.center(80)}")
    print("=" * 80 + "\n")

def test_environment():
    """Test if the Atari Boxing environment works"""
    try:
        # Try different environment IDs until one works
        env_ids = ["Boxing-v4", "BoxingNoFrameskip-v4", "ALE/Boxing-v5"]
        
        for env_id in env_ids:
            try:
                print(f"Trying environment ID: {env_id}")
                env = gym.make(env_id, render_mode="rgb_array")
                obs, info = env.reset()
                print(f"✓ Successfully created environment: {env_id}")
                print(f"  Observation shape: {obs.shape}")
                print(f"  Action space: {env.action_space}")
                
                # Take a few random actions
                frames = []
                total_reward = 0
                
                for _ in range(10):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    frames.append(env.render())
                    total_reward += reward
                    if terminated or truncated:
                        break
                
                env.close()
                print(f"  Performed 10 random actions, total reward: {total_reward}")
                print(f"  Captured {len(frames)} frames\n")
                
                # Save the working environment ID
                return env_id
            
            except Exception as e:
                print(f"× Failed to create {env_id}: {str(e)}\n")
    
    except Exception as e:
        print(f"Error testing environments: {str(e)}")
        return None

def quick_test(env_id, timesteps=1000):
    """Run a quick test of training and visualization"""
    print_header(f"QUICK TEST MODE: Training for {timesteps} timesteps")
    
    # Create directories
    os.makedirs("test_output", exist_ok=True)
    
    # Create and wrap the environment
    env = gym.make(env_id, render_mode="rgb_array")
    env = AtariWrapper(env)
    
    # Train a DQN agent for a short time
    print("Training a DQN agent...")
    model = DQN("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    
    # Evaluate the agent
    print("\nEvaluating agent...")
    
    # Save the model
    model_path = "test_output/test_model.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Run an evaluation episode
    obs, _ = env.reset(seed=42)
    total_reward = 0
    episode_steps = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        episode_steps += 1
        done = terminated or truncated
    
    print(f"Evaluation complete:")
    print(f"  Total reward: {total_reward}")
    print(f"  Episode steps: {episode_steps}")
    
    env.close()
    
    print("\nQuick test completed successfully!")
    return True

def run_training_mode(env_id, algorithm=None, timesteps=100000):
    """Run full training with specified algorithm(s)"""
    if algorithm is None or algorithm.lower() == "all":
        print_header("TRAINING ALL ALGORITHMS")
        results = compare_algorithms(env_id=env_id, timesteps=timesteps)
        return results
    
    print_header(f"TRAINING {algorithm.upper()}")
    
    if algorithm.upper() == "DQN":
        model, mean_reward, std_reward = train_and_evaluate(DQN, env_id, timesteps)
    elif algorithm.upper() == "A2C":
        model, mean_reward, std_reward = train_and_evaluate(A2C, env_id, timesteps)
    elif algorithm.upper() == "PPO":
        model, mean_reward, std_reward = train_and_evaluate(PPO, env_id, timesteps)
    else:
        print(f"Unknown algorithm: {algorithm}")
        return None
    
    print(f"\nTraining completed. Final performance: {mean_reward:.2f} ± {std_reward:.2f}")
    return {"mean_reward": mean_reward, "std_reward": std_reward}

def run_visualization_mode(env_id, algorithm=None, timesteps=100000, check_freq=5000):
    """Run training with visualization of progress"""
    print_header("TRAINING VISUALIZATION MODE")
    
    if algorithm is None or algorithm.lower() == "all":
        # Train all algorithms with visualization
        algorithms = [DQN, A2C, PPO]
        for algo in algorithms:
            visualize_training_process(
                algorithm=algo,
                total_timesteps=timesteps,
                check_freq=check_freq,
                env_id=env_id
            )
    else:
        # Train a specific algorithm
        if algorithm.upper() == "DQN":
            visualize_training_process(DQN, timesteps, check_freq, env_id)
        elif algorithm.upper() == "A2C":
            visualize_training_process(A2C, timesteps, check_freq, env_id)
        elif algorithm.upper() == "PPO":
            visualize_training_process(PPO, timesteps, check_freq, env_id)
        else:
            print(f"Unknown algorithm: {algorithm}")
            return None
    
    print("\nVisualization complete. Check the training_videos/ directory for results.")

def run_comparison_mode(env_id):
    """Run comparison of trained models"""
    print_header("MODEL COMPARISON MODE")
    
    # Check if models exist
    required_dirs = ["checkpoints", "models"]
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    if missing_dirs:
        print(f"Error: Missing directories: {', '.join(missing_dirs)}")
        print("Please train models first before running comparison mode.")
        return None
    
    # Run the comparison
    models, results = compare_models_from_checkpoints()
    
    print("\nComparison complete. Check the comparison_videos/ directory for results.")
    return results

def main():
    parser = argparse.ArgumentParser(description='Atari Boxing RL Experiments')
    
    # Main operation mode
    parser.add_argument('mode', type=str, choices=['test', 'train', 'visualize', 'compare', 'check'],
                        help='Operation mode: test (quick test), train (full training), ' +
                             'visualize (training with visualizations), compare (compare trained models), ' +
                             'check (only check if environment works)')
    
    # Optional arguments
    parser.add_argument('--algo', type=str, default=None, 
                        help='Algorithm to use: DQN, A2C, PPO, or "all" for all algorithms')
    
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Number of timesteps for training')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set default timesteps based on mode
    if args.timesteps is None:
        if args.mode == 'test':
            args.timesteps = 1000
        elif args.mode == 'train':
            args.timesteps = 100000
        elif args.mode == 'visualize':
            args.timesteps = 50000
    
    # First check if environment works and get the working environment ID
    if args.mode == 'check':
        working_env_id = test_environment()
        if working_env_id:
            print(f"\nEnvironment check passed! Use this ID in your experiments: {working_env_id}")
            return 0
        else:
            print("\nFailed to find a working environment ID. Please check your installation.")
            return 1
    
    # For other modes, find a working environment first
    print("Checking for a working Boxing environment...")
    working_env_id = test_environment()
    
    if not working_env_id:
        print("Error: Could not find a working Boxing environment.")
        print("Please check your installation of gymnasium and ale-py.")
        return 1
    
    # Run the selected mode
    if args.mode == 'test':
        quick_test(working_env_id, args.timesteps)
    
    elif args.mode == 'train':
        run_training_mode(working_env_id, args.algo, args.timesteps)
    
    elif args.mode == 'visualize':
        check_freq = min(args.timesteps // 4, 5000)  # Create at least 4 checkpoints
        run_visualization_mode(working_env_id, args.algo, args.timesteps, check_freq)
    
    elif args.mode == 'compare':
        run_comparison_mode(working_env_id)
    
    return 0

if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    elapsed_time = time.time() - start_time
    
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    sys.exit(exit_code) 