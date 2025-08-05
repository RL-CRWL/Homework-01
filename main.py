import Bandit as bd
import Plot as pt
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Multi-Armed Bandit Experiment")
    print("=" * 40)
    
    # Run comparison
    results = pt.compare_algorithms(arm_count=10, steps=1000, experiments=100)
    
    # Print final results
    print("\nFinal Average Rewards:")
    for alg_name, data in results.items():
        final_reward = data['avg_reward'][-1]
        final_optimal_pct = data['optimal_action_pct'][-100:].mean() * 100  # Last 100 steps
        print(f"{alg_name:20}: {final_reward[0]:.3f} (optimal: {final_optimal_pct:.1f}%)")
    
    # Plot results
    pt.plot_results(results)
    
    # Demonstrate single bandit usage
    print("\n" + "="*40)
    print("Single Bandit Demonstration:")
    
    bandit = bd.Bandit(count=5)
    print(f"True arm values: {[f'{arm.mean[0]:.2f}' for arm in bandit.arms]}")
    print(f"Optimal arm: {bandit.get_optimal_arm()}")
    
    # Test each algorithm briefly
    for name, alg in [("ε-greedy", bd.EpsilonGreedy(0.1)), ("UCB", bd.UCB(2.0))]:
        bandit.reset()
        print(f"\n{name} after 100 pulls:")
        for _ in range(100):
            arm = alg.select_arm(bandit)
            bandit.pull_arm(arm)
        
        print(f"  Estimated values: {bandit.estimated_values}")
        print(f"  Pull counts: {bandit.pull_count}")
        print(f"  Total reward: {bandit.total_reward[0]:.2f}")
        
value = 0
if value == 5:  
    def run_single_experiment(algorithm, n_arms=10, n_steps=1000):
        """Run a single experiment and return average reward"""
        bandit = bd.Bandit(n_arms)
        
        # Reset algorithm state if needed
        if hasattr(algorithm, 'initialized'):
            algorithm.initialized = False
        
        for _ in range(n_steps):
            arm_idx = algorithm.select_arm(bandit)
            bandit.pull_arm(arm_idx)
        
        # Return average reward over all steps
        return np.mean(bandit.rewards_history)
            
    def parameter_sweep(n_experiments=100, n_arms=10, n_steps=1000):
        """Run parameter sweep for all algorithms"""
        
        # Define parameter ranges (matching the x-axis values in your graph)
        # Note: these are powers of 2, so we use log scale
        param_values = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
        
        results = {}
        
        # ε-greedy parameter sweep
        print("Running ε-greedy parameter sweep...")
        epsilon_rewards = []
        for epsilon in param_values:
            if epsilon > 1:  # epsilon can't be > 1
                epsilon_rewards.append(np.nan)
                continue
                
            rewards = []
            for _ in range(n_experiments):
                alg = bd.EpsilonGreedy(epsilon=epsilon)
                avg_reward = run_single_experiment(alg, n_arms, n_steps)
                rewards.append(avg_reward)
            
            mean_reward = np.mean(rewards)
            epsilon_rewards.append(mean_reward)
            print(f"  ε={epsilon:.4f}: {mean_reward:.3f}")
        
        results['epsilon_greedy'] = epsilon_rewards
        
        # UCB parameter sweep  
        print("\nRunning UCB parameter sweep...")
        ucb_rewards = []
        for c in param_values:
            rewards = []
            for _ in range(n_experiments):
                alg = bd.UCB(c=c)
                avg_reward = run_single_experiment(alg, n_arms, n_steps)
                rewards.append(avg_reward)
            
            mean_reward = np.mean(rewards)
            ucb_rewards.append(mean_reward)
            print(f"  c={c:.4f}: {mean_reward:.3f}")
        
        results['ucb'] = ucb_rewards
        
        # Optimistic Greedy parameter sweep
        print("\nRunning Optimistic Greedy parameter sweep...")
        optimistic_rewards = []
        for q0 in param_values:
            rewards = []
            for _ in range(n_experiments):
                alg = bd.OptimisticGreedy(Q1=q0)
                avg_reward = run_single_experiment(alg, n_arms, n_steps)
                rewards.append(avg_reward)
            
            mean_reward = np.mean(rewards)
            optimistic_rewards.append(mean_reward)
            print(f"  Q₀={q0:.4f}: {mean_reward:.3f}")
        
        results['optimistic_greedy'] = optimistic_rewards
        
        return param_values, results

    def plot_parameter_comparison(param_values, results):
        """Create the parameter comparison plot"""
        plt.figure(figsize=(12, 8))
        
        # Plot each algorithm
        plt.plot(param_values, results['epsilon_greedy'], 'o-', color='red', 
                label='ε-greedy', linewidth=2, markersize=6)
        
        plt.plot(param_values, results['ucb'], 'o-', color='blue', 
                label='UCB', linewidth=2, markersize=6)
        
        plt.plot(param_values, results['optimistic_greedy'], 'o-', color='black', 
                label='greedy with\noptimistic\ninitialization', linewidth=2, markersize=6)
        
        # Set log scale for x-axis to match your graph
        plt.xscale('log', base=2)
        
        # Set x-axis ticks and labels
        plt.xticks(param_values, [f'1/{int(1/x)}' if x < 1 else str(int(x)) for x in param_values])
        
        # Labels and formatting
        plt.xlabel('ε / α / c / Q₀', fontsize=14)
        plt.ylabel('Average\nreward\nover first\n1000 steps', fontsize=14)
        plt.title('Summary comparison of algorithms', fontsize=16)
        
        # Set y-axis limits to match your graph
        plt.ylim(-2, 5)
        
        # Add legend
        plt.legend(fontsize=12, loc='upper right')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def quick_test():
        """Quick test with fewer experiments for faster results"""
        print("Running quick test (fewer experiments)...")
        param_values, results = parameter_sweep(n_experiments=20, n_steps=1000)
        plot_parameter_comparison(param_values, results)

    def full_experiment():
        """Full experiment matching the paper (takes longer)"""
        print("Running full experiment...")
        param_values, results = parameter_sweep(n_experiments=100, n_steps=1000)
        plot_parameter_comparison(param_values, results)

    # Run the experiment
    if __name__ == "__main__":
        print("Multi-Armed Bandit Parameter Comparison")
        print("=" * 50)
        print("Choose experiment type:")
        print("1. Quick test (20 runs per parameter) - ~30 seconds")
        print("2. Full experiment (2000 runs per parameter) - ~10 minutes")
        
        # For demonstration, run quick test
        # print("\nRunning quick test...")
        # quick_test()
        
        # Uncomment below for full experiment:
        full_experiment()