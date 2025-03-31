import gymnasium as gym
import ale_py  # Ensures ALE environments are registered
import matplotlib.pyplot as plt
from matplotlib import animation
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

# ---------------------------
# 1. TRAINING THE DQN AGENT
# ---------------------------

# Create the training environment with render_mode set to "rgb_array"
train_env = gym.make("ALE/Boxing-v5", render_mode="rgb_array")
# Wrap the environment for standard Atari preprocessing (frame stacking, resizing, etc.)
train_env = AtariWrapper(train_env)

# Initialize the DQN agent with the CNN policy for image-based observations
model = DQN("CnnPolicy", train_env, verbose=1)

# Train the agent (adjust total_timesteps as needed)
model.learn(total_timesteps=10000)

# ---------------------------
# 2. COLLECTING FRAMES FOR VISUALIZATION
# ---------------------------

# Create an evaluation environment in "rgb_array" mode to capture frames
eval_env = gym.make("ALE/Boxing-v5", render_mode="rgb_array")
eval_env = AtariWrapper(eval_env)
obs, info = eval_env.reset(seed=42)

frames = []  # List to store frames

terminated, truncated = False, False
while not (terminated or truncated):
    frames.append(obs)  # Save the current frame
    # Predict the best action using the trained model
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)

# Close environments once done
eval_env.close()
train_env.close()

# ---------------------------
# 3. CREATING THE ANIMATION
# ---------------------------
# Increase figure size and DPI for higher resolution
fig = plt.figure(figsize=(10, 8), dpi=150)
plt.axis('off')  # Hide axes for a cleaner display
im = plt.imshow(frames[0], interpolation="bilinear")  # Use bilinear interpolation for smoother upscaling

def update_frame(i):
    im.set_array(frames[i])
    return [im]

# Create the animation. Adjust 'interval' (in ms) as needed.
ani = animation.FuncAnimation(fig, update_frame, frames=len(frames), interval=50, blit=True)

# ---------------------------
# 4. SAVING THE ANIMATION
# ---------------------------
# Save the animation as an MP4 file with higher DPI.
output_filename = "boxing_agent_animation_highres.mp4"
ani.save(output_filename, writer="ffmpeg", fps=30, dpi=150)

print(f"Animation saved as {output_filename}")

plt.show()
