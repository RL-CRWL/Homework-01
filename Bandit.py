import numpy as np

class Arm:
    mean = 0
    std_dev = 0
    
    def __init__(self, mean=0, std_dev=3**0.5):
        self.mean = np.random.normal(loc=mean, scale=std_dev, size=1)
        self.std_dev = 1
        
    def get_reward(self):
        return np.random.normal(loc=self.mean, scale=self.std_dev, size=1)   

class Bandit:
    def __init__(self, arms, selection_algorithm):
        self.arms = arms
        self.selection_algorithm = selection_algorithm
        self.total_reward = 0.0
        self.pull_history = []

    def select_arm(self, step=None):
        if step is not None:
            return self.selection_algorithm.select_arm(self.arms, step)
        return self.selection_algorithm.select_arm(self.arms)

    def pull_arm(self, step=None):
        arm_index = self.select_arm(step)
        reward = float(self.arms[arm_index].get_reward())  
        self.total_reward += reward
        self.pull_history.append((arm_index, reward))
        return arm_index, reward

    def reset(self):
        self.total_reward = 0.0
        self.pull_history.clear()

    def get_total_reward(self):
        return self.total_reward

    def run(self, rounds, optimal_expected_value=None):
        rewards = []
        regret = []
        cumulative_reward = 0.0
        cumulative_regret = 0.0
        
        for t in range(1, rounds+1):
            _, reward = self.pull_arm(step=t)
            cumulative_reward += reward
            rewards.append(cumulative_reward / t)

            if optimal_expected_value is not None:
                cumulative_regret += (optimal_expected_value - reward)
                regret.append(cumulative_regret)
                
        return rewards, regret
