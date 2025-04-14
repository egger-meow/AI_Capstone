# Atari Boxing Reinforcement Learning Experiments

This project implements and compares different reinforcement learning algorithms on the Atari Boxing environment using Stable Baselines3 and Gymnasium.

## Project Structure

- `enc.py`: Basic DQN implementation with simple visualization
- `boxing_experiments.py`: Comprehensive implementation that trains and compares multiple RL algorithms
- `boxing_training_visualization.py`: Creates detailed visualizations of training progress over time
- `boxing_comparison.py`: Creates side-by-side comparisons of different trained models

## Setup and Requirements

1. Install the required packages:

```bash
pip install gymnasium[atari,accept-rom-license] stable-baselines3[extra] opencv-python matplotlib numpy
```

2. Make sure you have ffmpeg installed for video creation:

For Windows:
```
# Download from https://ffmpeg.org/download.html and add to PATH
```

For macOS:
```
brew install ffmpeg
```

For Linux:
```
apt-get install ffmpeg
```

## How to Use the Scripts

### 1. Basic DQN Training

```bash
python enc.py
```

This will train a simple DQN agent for 10,000 steps and create a basic animation.

### 2. Experiment with Multiple Algorithms

```bash
python boxing_experiments.py
```

This script will:
- Train DQN, A2C, and PPO algorithms
- Save models periodically
- Generate videos showing agent performance
- Create comparison charts and statistics

### 3. Visualize Training Progress

```bash
python boxing_training_visualization.py
```

This script will:
- Train models and save checkpoints at regular intervals
- Record videos at different stages of training to visualize learning progress
- Create learning curve animations showing reward improvement over time

### 4. Compare Trained Models

```bash
python boxing_comparison.py
```

This script will:
- Load previously trained models
- Create side-by-side comparison videos of different algorithms
- Generate performance metrics and comparison statistics

## Directory Structure (Created by the Scripts)

- `models/`: Saved model checkpoints for each algorithm
- `logs/`: Training logs and TensorBoard data
- `videos/`: Individual performance videos for each algorithm
- `checkpoints/`: Model checkpoints saved during training
- `training_videos/`: Videos showing progress at different training stages
- `training_stats/`: Learning curve plots and other statistics
- `comparison_videos/`: Side-by-side comparisons and performance metrics

## Notes on Hyperparameters

Each algorithm uses default hyperparameters that work reasonably well for Atari environments, but you may want to experiment with:

- **Learning Rates**: DQN (1e-4), A2C (7e-4), PPO (2.5e-4)
- **Buffer Size** (DQN): 100,000 transitions
- **Exploration Rate** (DQN): 10% initial exploration
- **Training Steps**: Default is 500,000 steps, which you can adjust based on your computational resources

## Customizing Training

To adjust training parameters, modify the following constants:
- In `boxing_experiments.py`: `TOTAL_TIMESTEPS`
- In `boxing_training_visualization.py`: `TOTAL_TIMESTEPS` and `CHECK_FREQ`

## References

- [Gymnasium Boxing Environment](https://www.gymlibrary.dev/environments/atari/boxing/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Boxing-RL GitHub Repository](https://github.com/rohilG/Boxing-RL) 