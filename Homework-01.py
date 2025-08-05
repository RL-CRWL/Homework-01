# Contributors
# 2541693   Wendy Maboa
# 2602515   Taboka Dube
# 2596852   Liam Brady
# 2333776   Refiloe Mopeloa

import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k=10):
        self.k = k
        self.true_means = np.random.normal(0, np.sqrt(3), k)
        self.variance = 1

    def pull(self, arm):
        return np.random.normal(self.true_means[arm], np.sqrt(self.variance))


class EpsilonGreedy:
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)  
        self.N = np.zeros(k)  

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.Q)

    def update(self, arm, reward):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


class OptimisticGreedy:
    def __init__(self, k, Q1):
        self.k = k
        self.Q = np.ones(k) * Q1  
        self.N = np.zeros(k)

    def select_arm(self):
        return np.argmax(self.Q)

    def update(self, arm, reward):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


class UCB:
    def __init__(self, k, c):
        self.k = k
        self.c = c
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.time = 0

    def select_arm(self):
        self.time += 1
        for i in range(self.k):
            if self.N[i] == 0:
                return i 
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.time) / self.N)
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


def run_algorithm(algo_class, bandit_class, steps, runs, **kwargs):
    rewards = np.zeros(steps)

    for _ in range(runs):
        bandit = bandit_class()
        agent = algo_class(bandit.k, **kwargs)
        total_reward = []

        for t in range(steps):
            arm = agent.select_arm()
            reward = bandit.pull(arm)
            agent.update(arm, reward)
            total_reward.append(reward)

        rewards += np.array(total_reward)

    return rewards / runs


np.random.seed(42)

steps = 1000
runs = 100
k = 10

reward_eps_greedy = run_algorithm(EpsilonGreedy, Bandit, steps, runs, epsilon=0.1)
reward_opt_greedy = run_algorithm(OptimisticGreedy, Bandit, steps, runs, Q1=5)
reward_ucb = run_algorithm(UCB, Bandit, steps, runs, c=2)

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# First subplot: Average reward comparison over time
ax1.plot(reward_eps_greedy, label='ε-greedy (ε=0.1)', alpha=0.8)
ax1.plot(reward_opt_greedy, label='Optimistic Greedy (Q1=5)', alpha=0.8)
ax1.plot(reward_ucb, label='UCB (c=2)', alpha=0.8)
ax1.set_xlabel('Steps')
ax1.set_ylabel('Average Reward')
ax1.set_title('Multi-Armed Bandit: Average Reward Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Hyperparameter analysis - wider range on log scale
epsilon_values = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1]
Q1_values = [1/8, 1/4, 1/2, 1, 2, 4, 8, 16]
c_values = [1/8, 1/4, 1/2, 1, 2, 4, 8]

eps_total_rewards = []
opt_total_rewards = []
ucb_total_rewards = []

for eps in epsilon_values:
    rewards = run_algorithm(EpsilonGreedy, Bandit, steps, runs, epsilon=eps)
    eps_total_rewards.append(np.sum(rewards))

for q1 in Q1_values:
    rewards = run_algorithm(OptimisticGreedy, Bandit, steps, runs, Q1=q1)
    opt_total_rewards.append(np.sum(rewards))

for c in c_values:
    rewards = run_algorithm(UCB, Bandit, steps, runs, c=c)
    ucb_total_rewards.append(np.sum(rewards))

# Second subplot: Hyperparameter sensitivity
ax2.plot(epsilon_values, eps_total_rewards, '-', label='ε-greedy', linewidth=2)
ax2.plot(Q1_values, opt_total_rewards, '-', label='Optimistic Greedy', linewidth=2)
ax2.plot(c_values, ucb_total_rewards, '-', label='UCB', linewidth=2)

ax2.set_xlabel('Parameter Value (ε / Q₁ / c)')
ax2.set_ylabel('Total Reward (1000 steps)')
ax2.set_title('Hyperparameter Sensitivity Analysis')
ax2.set_xscale('log')
ax2.set_xticks([1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16])
ax2.set_xticklabels(['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4', '8', '16'])
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()