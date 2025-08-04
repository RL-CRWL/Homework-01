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
        #not sure what this is i am using a medium article
        #dusting
        #PLEASEEEEEEEE lmfao
        self.selection_algorithm = selection_algorithm
        self.total_reward = 0.0
        self.pull_history = []
    
    def select_arm(self, step=None):
        if step is not None:
            return self.selection_algorithm.select_arm(self.arms, step)
        return self.selection_algorithm.select_arm(self.arms)

    
    def pull_Arm(self, step=None):
        arm_in = self.select_arm(step)
        reward = self.arms[arm_in].pull()
        self.total_reward = self.total_reward + reward
        self.pull_history.append((arm_in, reward))
        return arm_in, reward

    def reset(self):
        self.total_reward = 0.0
        self.pull_history.clear()
        for arm in self.arms:
            arm.reset()
    
    def get_total_reward(self):
        return self.total_reward
    
    def get_arm_stats(self):
        return[
            {
                'empirical_mean': arm.get_empirical_mean(),
                'pull_count': arm.get_pull_count()
            }
            for arm in self.arms
        ]

    def run(self, rounds, optimal_expected_value=None):
        reward  =     