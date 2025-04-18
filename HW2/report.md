# HW2 109700046

## Code Repository
**GitHub Link**: [https://github.com/egger-meow/AI_Capstone/tree/main/HW2](https://github.com/egger-meow/AI_Capstone/tree/main/HW2)

## Task I - Boxing (Atari)

### 1. Research Motivation

I have personally learned boxing for a while, so I have a deep connection with the content of this game. In boxing matches, it's not just about simple striking, but also involves complex strategies, timing, and understanding the opponent's psychology. It's fascinating to see how AI can learn to play a boxing game, as it tests not only the AI's reaction ability but also its capability to learn strategies.

In reinforcement learning, Boxing is an excellent test environment because:
- It has clear reward signals (scoring points by hitting the opponent)
- It has multiple possible actions (movement, punching, etc.)
- It requires strategic thinking (dodging opponent attacks, choosing the right moment to strike)
- It has competitive elements (fighting against computer AI)

### 2. Game Introduction

Boxing is a classic Atari game where the player controls a blue boxer fighting against a computer-controlled red boxer in a boxing ring.

**Game Environment Features**:
- **Action Space**: Discrete action space with 18 different actions, including:
  - Movement in different directions (up, down, left, right, diagonal)
  - Different height and angle punching actions
  - Defensive stance
  - No action
  
- **State Space**: Game screen with resolution of 210×160 pixels, 3 color channels (RGB). The AI needs to learn the game state from raw pixels.

- **Reward Mechanism**: 
  - +1 point for hitting the opponent
  - -1 point for being hit by the opponent
  - Fixed game duration, final score is the cumulative points

- **Termination Conditions**: Game ends after a fixed number of rounds or time

In the Boxing game, strategy is crucial. Players need to find the right moment to attack while avoiding being hit by the opponent. This is a zero-sum game where the total score is the difference between the two players' scores.

### 3. Implementation Pipeline

1. **Environment Setup**:
   - Load Boxing-v4 environment using OpenAI Gym and Stable-Baselines3
   - Apply preprocessing wrappers including frame stacking, grayscale, and normalization
   - Set random seeds to ensure experiment reproducibility

2. **Model Training**:
   - Create and configure models for each algorithm (DQN, A2C, PPO)
   - Set up network architecture (typically CNN + fully connected layers)
   - Use callback functions to save best models and record training statistics
   - Train DQN for 500,000 steps, A2C and PPO for 100,000 steps

3. **Model Evaluation**:
   - Load trained models
   - Test model performance across multiple episodes
   - Record rewards, episode lengths, and other statistics
   - Create comparison videos to showcase different algorithms' performance

4. **Result Visualization**:
   - Plot learning curves to show training progress
   - Perform multi-model comparison analysis

### 4. Algorithm Introduction

#### 4.1 DQN (Deep Q-Network)
DQN combines deep learning with Q-learning to handle high-dimensional state spaces. It uses a neural network to approximate Q-values and employs experience replay to break sample correlations. The target network helps stabilize training by providing fixed targets for TD learning.

#### 4.2 A2C (Advantage Actor-Critic)
A2C uses two networks: an Actor for action selection and a Critic for state evaluation. The advantage function A(s,a) = Q(s,a) - V(s) reduces variance in policy updates. It collects experience synchronously across multiple environments.

#### 4.3 PPO (Proximal Policy Optimization)
PPO is a policy gradient method that prevents large policy updates through a clipped objective function. It performs multiple training rounds on the same data batch, improving sample efficiency while maintaining training stability.

### 5. Results

I tested in two modes:
1. **Training Mode**: DQN trained for 500,000 steps, A2C and PPO trained for 100,000 steps
2. **Visualization Mode**: All algorithms evaluated for 50,000 steps

#### 5.1 Training Mode Results:

| Algorithm | Mean Reward | Min/Max | Knockouts | Avg. Length |
|-----------|------------|---------|-----------|-------------|
| DQN       | 5.6 ± 6.5  | -6 / 18 | 0/10 (0%) | 440.9       |
| A2C       | -27.3 ± 8.0| -40 / -17| 0/10 (0%) | 440.9       |
| PPO       | 6.2 ± 7.8  | -4 / 23 | 0/10 (0%) | 440.9       |

#### 5.2 Visualization Mode Results:

| Algorithm | Mean Reward | Min/Max | Knockouts | Avg. Length |
|-----------|------------|---------|-----------|-------------|
| DQN       | -2.3 ± 5.2 | -10 / 7 | 0/10 (0%) | 440.9       |
| A2C       | -27.3 ± 8.0| -40 / -17| 0/10 (0%) | 440.9       |
| PPO       | 6.0 ± 7.0  | -5 / 16 | 0/10 (0%) | 440.9       |

#### 5.3 Learning Curve Analysis:

From the training process learning curves, we can observe:
- **PPO**: Learned the fastest and converged most stably
- **DQN**: Learned more slowly but eventually achieved good results
- **A2C**: Performed worst, failing to learn effective strategies

### 6. Discussion

Through this project, I gained several important insights about reinforcement learning algorithms and their application to the Boxing game:

1. **Algorithm Performance Insights**:
   - PPO's superior performance in this task demonstrates the importance of stable policy updates in adversarial environments
   - DQN's requirement for more training steps highlights the trade-off between sample efficiency and final performance
   - A2C's poor performance suggests that some algorithms may be more sensitive to environment characteristics

2. **Key Learnings**:
   - Algorithm choice significantly impacts performance, even with identical training conditions
   - The relationship between training time and performance is not linear across different algorithms
   - Environment characteristics (like action space complexity) can dramatically affect algorithm performance

3. **Remaining Questions**:
   - Why did A2C perform so poorly despite its theoretical advantages?
   - Could different hyperparameter settings significantly improve A2C's performance?
   - How would the algorithms perform with different reward shaping approaches?
   - What is the minimum training time needed for each algorithm to achieve stable performance?

4. **Future Directions**:
   - Investigate the impact of different network architectures on algorithm performance
   - Explore the relationship between training time and performance in more detail
   - Study the effect of different reward shaping strategies
   - Analyze the role of exploration strategies in algorithm performance

This project has deepened my understanding of reinforcement learning algorithms and their practical applications. The results highlight the importance of careful algorithm selection and the need for thorough experimentation when applying these methods to real-world problems.

## Task II - Crytpo Trading agent

### motivation: 
personaly, I am a cryto trader who trade on the app OKX, I had tried auto trading with my own code before but my algorithm was too bad to resulting yeilds.


