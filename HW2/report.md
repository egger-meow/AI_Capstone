# HW2 109700046

## Task I - Boxing (Atari)

### motivation: 
personaly, I have learned boxing for a while, so I have quite a deep connection to the content of the game  

### game introduction 
(action space,states explain u need to finish):

### algorithms
DQN
A2C
PPO


train result (10000 timesteps) evaluation mode:



================================================================================
                             MODEL COMPARISON MODE
================================================================================


Found 3 model checkpoints:
  DQN: ./models/DQN/best_model.zip
  A2C: ./models/A2C/best_model.zip
  PPO: ./models/PPO/best_model.zip

Creating side-by-side comparison video...
Recording episode 1/3
Episode 1 completed. Final scores:
  DQN: 19.0
  A2C: -29.0
  PPO: 5.0
Recording episode 2/3
Episode 2 completed. Final scores:
  DQN: 10.0
  A2C: -17.0
  PPO: -2.0
Recording episode 3/3
Episode 3 completed. Final scores:
  DQN: 4.0
  A2C: -29.0
  PPO: -5.0
Comparison video saved to comparison_videos/algorithm_comparison.mp4

Evaluating model performance...
  A2C: -17.0
  PPO: -2.0
Recording episode 3/3
Episode 3 completed. Final scores:
  DQN: 4.0
  A2C: -29.0
  PPO: -5.0
Comparison video saved to comparison_videos/algorithm_comparison.mp4

Evaluating model performance...
  PPO: -2.0
Recording episode 3/3
Episode 3 completed. Final scores:
  DQN: 4.0
  A2C: -29.0
  PPO: -5.0
Comparison video saved to comparison_videos/algorithm_comparison.mp4

Evaluating model performance...
Episode 3 completed. Final scores:
  DQN: 4.0
  A2C: -29.0
  PPO: -5.0
Comparison video saved to comparison_videos/algorithm_comparison.mp4

Evaluating model performance...
  A2C: -29.0
  PPO: -5.0
Comparison video saved to comparison_videos/algorithm_comparison.mp4

Evaluating model performance...
Comparison video saved to comparison_videos/algorithm_comparison.mp4

Evaluating model performance...

Evaluating model performance...
Evaluating DQN...
Evaluating A2C...
Evaluating PPO...

===== Model Performance Comparison =====
| Algorithm | Mean Reward | Min/Max | Knockouts | Avg. Length |
Evaluating DQN...
Evaluating A2C...
Evaluating PPO...

===== Model Performance Comparison =====
| Algorithm | Mean Reward | Min/Max | Knockouts | Avg. Length |

===== Model Performance Comparison =====
| Algorithm | Mean Reward | Min/Max | Knockouts | Avg. Length |
|-----------|-------------|---------|-----------|-------------|
| DQN | 5.6 ± 6.5 | -6 / 18 | 0/10 (0%) | 440.9 |
| A2C | -27.3 ± 8.0 | -40 / -17 | 0/10 (0%) | 440.9 |
|-----------|-------------|---------|-----------|-------------|
| DQN | 5.6 ± 6.5 | -6 / 18 | 0/10 (0%) | 440.9 |
| A2C | -27.3 ± 8.0 | -40 / -17 | 0/10 (0%) | 440.9 |
| DQN | 5.6 ± 6.5 | -6 / 18 | 0/10 (0%) | 440.9 |
| A2C | -27.3 ± 8.0 | -40 / -17 | 0/10 (0%) | 440.9 |
| A2C | -27.3 ± 8.0 | -40 / -17 | 0/10 (0%) | 440.9 |
| PPO | 6.2 ± 7.8 | -4 / 23 | 0/10 (0%) | 440.9 |

Comparison completed! Results and videos are available in the comparison_videos/ directory.

Comparison complete. Check the comparison_videos/ directory for results.

Total execution time: 90.28 seconds



## Task II - Crytpo Trading agent

### motivation: 
personaly, I am a cryto trader who trade on the app OKX, I had tried auto trading with my own code before but my algorithm was too bad to resulting yeilds.


