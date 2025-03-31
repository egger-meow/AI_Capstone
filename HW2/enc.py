import gymnasium as gym
import ale_py  # Ensures ALE environments are registered
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Create the Boxing environment with the human render mode disabled during training
env = gym.make("ALE/Boxing-v5", render_mode="rgb_array")
# Optionally wrap the environment for preprocessing (frame stacking, grayscale, etc.)
env = AtariWrapper(env)

# Initialize the DQN agent using a convolutional neural network policy (CnnPolicy)
model = DQN("CnnPolicy", env, verbose=1)

# Train the agent for a specified number of timesteps (adjust as needed)
model.learn(total_timesteps=10000)

# To see the trained agent in action, create a new environment with rendering enabled
eval_env = gym.make("ALE/Boxing-v5", render_mode="human")
eval_env = AtariWrapper(eval_env)
obs, info = eval_env.reset()

# Run a short evaluation loop to display agent performance
done = False
while not done:
    # Use the trained policy to predict the best action
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    if terminated or truncated:
        obs, info = eval_env.reset()
        done = True  # For this demo, we end after one episode

eval_env.close()
env.close()
