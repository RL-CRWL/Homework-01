import numpy as np

class Bandit:
    def __init__(self,k=10):
        self.k=k
        self.true_means=np.random.normal(0,np.sqrt(3),k)
        self.variance = 1

    def pull(self, arm):
        return np.random.normal(self.means[arm], np.sqrt(self.variance))
    


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


def run ...