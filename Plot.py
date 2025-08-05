import Bandit as bd
import numpy as np
import matplotlib.pyplot as plt

def run_experiment(bandit:bd.Bandit, algorithm, steps=1000):
    bandit.reset()
    
    # initialization for Optimistic Greedy
    algorithm.initialized = False if hasattr(algorithm, 'initialized') else None
    
    # run the experiment for steps
    for _ in range(steps):
        arm_id = algorithm.select_arm(bandit)
        bandit.pull_arm(arm_id)
        
    return bandit.rewards_history, bandit.actions_history

def compare_algorithms(arm_count = 10, steps = 1000, experiments = 100):
    algorithms = {
        "EpsilonGreedy (epsilon = 0.1)": bd.EpsilonGreedy(),
        "OptimisticGreedy (Q1 = 5)": bd.OptimisticGreedy(),
        "UCB (c = 2)": bd.UCB()        
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"Running {name}...")
        all_rewards = []
        all_optimal_actions = []
        
        for experiment in range(experiments):
            # Create new bandit for each experiment
            bandit = bd.Bandit(arm_count)
            optimal_arm = bandit.get_optimal_arm()
            
            rewards, actions = run_experiment(bandit, algorithm, steps)
            
            # Calculate cumulative average reward     # this creates a list of numbers from 1 to the number of actions, adding the previous number to the next
            cumulative_rewards = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
            all_rewards.append(rewards)
            
            # Track optimal action percentage
            optimal_actions = [1 if action == optimal_arm else 0 for action in actions]
            all_optimal_actions.append(optimal_actions)
            
         # Average across experiments
        avg_rewards = np.mean(all_rewards, axis=0)
        avg_optimal = np.mean(all_optimal_actions, axis=0)
        
        results[name] = {
            'avg_reward': avg_rewards,
            'optimal_action_pct': avg_optimal
        }
        
    return results
    
def plot_results(results):
    """Plot comparison of algorithms"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot average reward
    for alg_name, data in results.items():
        ax1.plot(data['avg_reward'], label=alg_name)
    
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Average Reward Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot optimal action percentage
    for alg_name, data in results.items():
        # Smooth the optimal action percentage with moving average
        window = 50
        smoothed = np.convolve(data['optimal_action_pct'], np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(data['optimal_action_pct'])), smoothed, label=alg_name)
    
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('% Optimal Action')
    ax2.set_title('Optimal Action Percentage Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()