import numpy as np
import gymnasium as gym
import random

# 1. Setup Environment (4x4 Frozen Lake)
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

# 2. Initialize Q-Table with zeros
state_size = env.observation_space.n
action_size = env.action_space.n
qtable = np.zeros((state_size, action_size))

# 3. Model Hyperparameters
total_episodes = 2000
learning_rate = 0.7
gamma = 0.95
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

print("Starting Training on ASUS TUF...")

# 4. Training Loop
for episode in range(total_episodes):
    state = env.reset()[0]
    step = 0
    done = False
    
    for step in range(100):
        # Epsilon-Greedy Policy: Exploration vs Exploitation[cite: 3]
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(qtable[state])
        else:
            action = env.action_space.sample()

        # Take action and observe outcome[cite: 3]
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Update Q-Value using Bellman Equation[cite: 3]
        qtable[state, action] = qtable[state, action] + learning_rate * (
            reward + gamma * np.max(qtable[new_state]) - qtable[state, action]
        )
        
        state = new_state
        if done: break
        
    # Reduce epsilon to decrease exploration over time[cite: 3]
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    if (episode + 1) % 200 == 0:
        print(f"Episode {episode + 1}/{total_episodes} completed.")

print("\nTraining Finished!")

# 5. Evaluate the Agent[cite: 3]
print("\nTesting the trained Agent:")
state = env.reset()[0]
total_rewards = 0

for step in range(100):
    action = np.argmax(qtable[state]) # Greedy Policy for testing[cite: 3]
    new_state, reward, terminated, truncated, info = env.step(action)
    state = new_state
    if terminated or truncated:
        if reward == 1:
            print("Success! Agent reached the Goal.")
        else:
            print("Failed! Agent fell into a Hole.")
        break

env.close()