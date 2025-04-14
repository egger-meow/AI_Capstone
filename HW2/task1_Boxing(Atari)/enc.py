import gymnasium as gym
import ale_py  # Ensures ALE environments are registered
import matplotlib.pyplot as plt
from matplotlib import animation
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
import os

# Create directories
os.makedirs("videos", exist_ok=True)

# ---------------------------
# 0. FIND WORKING ENVIRONMENT
# ---------------------------

# Try different environment IDs
env_ids = ["Boxing-v4", "BoxingNoFrameskip-v4", "ALE/Boxing-v5"]
working_env_id = None

for env_id in env_ids:
    try:
        print(f"Trying environment ID: {env_id}")
        test_env = gym.make(env_id, render_mode="rgb_array")
        test_obs, test_info = test_env.reset()
        test_env.close()
        working_env_id = env_id
        print(f"✓ Successfully created environment: {env_id}")
        break
    except Exception as e:
        print(f"× Failed to create {env_id}: {str(e)}")

if working_env_id is None:
    print("Error: Could not find a working Boxing environment.")
    print("Please check your installation of gymnasium and ale-py.")
    exit(1)

# ---------------------------
# 1. TRAINING THE DQN AGENT
# ---------------------------

# Set total training timesteps
TOTAL_TIMESTEPS = 10000
print(f"\nTraining DQN agent for {TOTAL_TIMESTEPS} timesteps...\n")

# Create the training environment with render_mode set to "rgb_array"
train_env = gym.make(working_env_id, render_mode="rgb_array")
# Wrap the environment for standard Atari preprocessing (frame stacking, resizing, etc.)
train_env = AtariWrapper(train_env)

# Initialize the DQN agent with the CNN policy for image-based observations
model = DQN("CnnPolicy", train_env, verbose=1)

# Train the agent (adjust total_timesteps as needed)
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# ---------------------------
# 2. COLLECTING FRAMES FOR VISUALIZATION
# ---------------------------

# Create an evaluation environment in "rgb_array" mode to capture frames
eval_env = gym.make(working_env_id, render_mode="rgb_array")
eval_env = AtariWrapper(eval_env)
obs, info = eval_env.reset(seed=42)

frames = []  # List to store frames
rewards = []  # List to store rewards
total_reward = 0  # Track total reward

terminated, truncated = False, False
while not (terminated or truncated):
    frames.append(eval_env.render())  # Save the current frame
    # Predict the best action using the trained model
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    
    # Track rewards
    total_reward += reward
    rewards.append(total_reward)

# Close environments once done
eval_env.close()
train_env.close()

print(f"\nEvaluation complete. Total reward: {total_reward}")
print(f"Captured {len(frames)} frames")

# ---------------------------
# 3. CREATING THE ANIMATION
# ---------------------------
# Create a figure with two subplots: game frames and reward plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), 
                              gridspec_kw={'height_ratios': [3, 1]})

# Game frames display
ax1.axis('off')  # Hide axes for a cleaner display
im = ax1.imshow(frames[0], interpolation="bilinear")  # Use bilinear interpolation for smoother upscaling
ax1.set_title(f"Boxing Agent (DQN) - Total Reward: {total_reward}", fontsize=16)

# Reward plot
reward_line, = ax2.plot([], [], 'b-', linewidth=2)
ax2.set_xlim(0, len(frames))
ax2.set_ylim(min(min(rewards), 0) - 1, max(rewards) + 1)
ax2.set_xlabel("Steps", fontsize=14)
ax2.set_ylabel("Cumulative Reward", fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7)

def init():
    """Initialize the animation"""
    reward_line.set_data([], [])
    return [im, reward_line]

def update_frame(i):
    """Update function for animation"""
    # Update the game frame
    im.set_array(frames[i])
    
    # Update the reward plot
    reward_line.set_data(range(i+1), rewards[:i+1])
    
    return [im, reward_line]

# Create the animation with both game frames and reward plot
ani = animation.FuncAnimation(fig, update_frame, frames=len(frames), 
                             init_func=init, interval=50, blit=True)

# ---------------------------
# 4. SAVING THE ANIMATION
# ---------------------------
# Save the animation as an MP4 file with higher DPI
output_filename = "videos/boxing_agent_animation.mp4"
ani.save(output_filename, writer="ffmpeg", fps=30, dpi=150)

print(f"Animation saved as {output_filename}")

plt.show()
