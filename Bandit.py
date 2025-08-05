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
    arms:list = []
    count:int = 0
    total_reward:float = 0
    pull_count = []
    arm_id:int = 0
    
    def __init__(self, count=10):
        self.count = count
        for i in range(self.count):
            self.arms.append(Arm())
        self.reset()
        
    def reset(self):
        self.total_reward = 0
        self.pull_count = np.zeros(self.count)
        self.estimated_values = np.zeros(self.count)
        self.rewards_history = []
        self.actions_history = []
        
    def pull_arm(self, arm_id):
        # obtain the current arms reward
        reward = self.arms[arm_id].get_reward()
        
        # keep track of the number of times this arm is pulled
        self.pull_count[arm_id] += 1
        
        # add the current reward to the total reward
        self.total_reward += reward
        
        # Update Q: Qn+1 = Qn + (Rn - Qn)/n
        self.estimated_values[arm_id] += (reward - self.estimated_values[arm_id])/self.pull_count[arm_id]
        
        # keep track of the reward from the taken action at this time step
        self.rewards_history.append(reward)
        
        # keep track of the action that was taken at each time step
        self.actions_history.append(arm_id)
        
        return reward
    
    # this function wll help when we need to calculate optimal action percentage
    def get_optimal_arm(self):
        # Return the argmax of the arm that gives the highest mean
        return np.argmax([arm.mean for arm in self.arms])
    
class EpsilonGreedy:
    epsilon = 0
    
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        
    def select_arm(self, bandit:Bandit):
        # check if arm should explore or exploit
        if np.random.random() < self.epsilon:
            # explore
            return np.random.randint(bandit.count)
        else:
            # exploit
            return np.argmax(bandit.estimated_values)
        
class OptimisticGreedy:
    Q1 = 0
    initialized = False
    def __init__(self, Q1=5.0):
        self.Q1 = Q1
        
    def select_arm(self, bandit:Bandit):
        # check if the optimistic value has been initialized, that is, has the estimated values for all arms been set to Q1
        if not self.initialized:
            bandit.estimated_values.fill(self.Q1)
            self.initialized = True
            
        # choose the highest estimate of every subsequent value throughout the computation of the estimate
        return np.argmax(bandit.estimated_values)
    
class UCB:
    c:float = 0
    
    def __init__(self, c:float = 2):
        self.c = c
    
    def select_arm(self, bandit:Bandit):
        # obtain which "time step" we are at right now
        total_pulls = np.sum(bandit.pull_count)
        
        # unpulled arms should always take priority
        unpulled_arms = np.where(bandit.pull_count == 0)[0]
        
        # this check is important because if an arm is unpulled then the pull count is 0 and this will result in a division of 0
        if len(unpulled_arms) > 0:
            # if any arms are unpulled return the first unpulled arms index
            return unpulled_arms[0]
        
        ucb_values = bandit.estimated_values + self.c * np.sqrt(np.log(total_pulls) / bandit.pull_count)
        
        # return the index of the arm which results in the highest ucb value
        return np.argmax(ucb_values)