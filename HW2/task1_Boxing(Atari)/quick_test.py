import os
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Create output directory
os.makedirs("quick_test_output", exist_ok=True)

# Try different environment IDs
env_ids = ["Boxing-v4", "BoxingNoFrameskip-v4", "ALE/Boxing-v5"]
working_env = None

for env_id in env_ids:
    try:
        print(f"Trying to create environment: {env_id}")
        env = gym.make(env_id, render_mode="rgb_array")
        obs, info = env.reset()
        print(f"✓ Successfully created environment: {env_id}")
        working_env = env
        break
    except Exception as e:
        print(f"× Failed to create {env_id}: {str(e)}")

if working_env is None:
    print("Failed to create any Boxing environment.")
    exit(1)

# Wrap the environment
env = AtariWrapper(working_env)

print(f"Training a DQN agent for a short time (1000 steps)...")
model = DQN("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

print("Running evaluation...")
obs, info = env.reset(seed=42)
frames = []
rewards = []
total_reward = 0

done = False
while not done:
    # Render and save the frame
    frame = env.render()
    frames.append(frame)
    
    # Get action from model
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Track rewards
    total_reward += reward
    rewards.append(total_reward)
    
    done = terminated or truncated

# Close the environment
env.close()

print(f"Evaluation complete! Total reward: {total_reward}")
print(f"Collected {len(frames)} frames")

# Create a simple animation with reward plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10),
                              gridspec_kw={'height_ratios': [3, 1]})

# Display the first frame
frame_img = ax1.imshow(frames[0])
ax1.axis('off')
ax1.set_title("Boxing Agent Performance")

# Create a reward plot
reward_line, = ax2.plot([], [], 'b-')
ax2.set_xlim(0, len(frames))
ax2.set_ylim(min(min(rewards), 0) - 1, max(rewards) + 1)
ax2.set_xlabel("Steps")
ax2.set_ylabel("Cumulative Reward")
ax2.grid(True)

def init():
    reward_line.set_data([], [])
    return [frame_img, reward_line]

def update(i):
    frame_img.set_array(frames[i])
    reward_line.set_data(range(i+1), rewards[:i+1])
    return [frame_img, reward_line]

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(frames),
                             init_func=init, blit=True, interval=50)

# Save animation
output_file = "quick_test_output/boxing_test_animation.mp4"
print(f"Saving animation to {output_file}...")
ani.save(output_file, writer="ffmpeg", fps=30)

print(f"Done! Animation saved to {output_file}")

# Display a static frame with final score for reference
plt.figure(figsize=(10, 8))
plt.imshow(frames[-1])
plt.axis('off')
plt.title(f"Final Frame - Score: {total_reward}", fontsize=14)
plt.savefig("quick_test_output/final_frame.png", dpi=150)
plt.close() 