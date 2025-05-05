import argparse
import numpy as np
import matplotlib.pyplot as plt
from algorithms import policy_iteration, q_learning, sarsa_lambda, value_iteration
from constants import K, T, PRICES, TRACE_DECAY, GAUSSIAN_SMOOTHING_STD
from scipy.ndimage import gaussian_filter1d

def plot_policy(policy, title):
    """ Plot the pricing policy as a heatmap. """

    policy_matrix = np.zeros((K, T))
    for inv in range(1, K + 1):
        for t in range(1, T + 1):
            policy_matrix[inv - 1, t - 1] = policy[(inv, t)]

    plt.figure()
    plt.imshow(policy_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Price')
    plt.title(title)
    plt.ylabel('Remaining Seats')
    plt.xlabel('Days Left')
    plt.ylim(1, K - 1)
    plt.xlim(1, T - 1)
    plt.show()

def plot_metrics(metrics, title):
    """ Plot evolution of different metrics for RL algorithms. """

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)

    axs[0, 0].plot(gaussian_filter1d(metrics['episode_rewards'], sigma=GAUSSIAN_SMOOTHING_STD))
    axs[0, 0].set_title('Total Discounted Rewards per Episode')
    axs[0, 0].set_ylabel('Total Discounted Reward')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].grid(linewidth=0.35)

    axs[0, 1].plot(gaussian_filter1d(metrics['episode_regrets'], sigma=GAUSSIAN_SMOOTHING_STD))
    axs[0, 1].set_title('Instantaneous Episodic Regret')
    axs[0, 1].set_ylabel('Regret')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].grid(linewidth=0.35)

    axs[1, 0].plot(metrics['cumulative_rewards'])
    axs[1, 0].set_title('Cumulative Discounted Rewards over Episodes')
    axs[1, 0].set_ylabel('Cumulative Discounted Reward')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].grid(linewidth=0.35)

    axs[1, 1].plot(metrics['cumulative_regrets'])
    axs[1, 1].set_title('Cumulative Regrets over Episodes')
    axs[1, 1].set_ylabel('Cumulative Regret')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].grid(linewidth=0.35)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Handle command line arguments
    parser = argparse.ArgumentParser(description="Tabular RL for Finite Inventory Pricing.")
    parser.add_argument("--verbose", action="store_true", \
                help="Print progress after every iteration.")
    parser.add_argument("--method", type=str, default="all", \
                choices=["all", "value-iteration", "policy-iteration", "q-learning", "sarsa"], \
                help="Algorithm to choose for solving the MDP.")
    args = parser.parse_args()

    # Call algorithms for solving MDP
    if args.method in ["all", "value-iteration"]:
        V_vi, policy_vi = value_iteration(verbose=args.verbose)
        optimal_value = V_vi[(K, T)]
        print(f"Finished Value Iteration with optimal value {optimal_value:.2f}")
        plot_policy(policy_vi, title="Value Iteration: Optimal Pricing Policy")

    if args.method in ["all", "policy-iteration"]:
        V_pi, policy_pi = policy_iteration(verbose=args.verbose)
        optimal_value = V_pi[(K, T)]
        print(f"Finished Policy Iteration with optimal value {optimal_value:.2f}")
        plot_policy(policy_pi, title="Policy Iteration: Optimal Pricing Policy")

    if args.method in ["all", "q-learning"]:
        Q_ql, policy_ql, metrics_ql = q_learning(verbose=args.verbose)
        optimal_value = max([ Q_ql[(K, T, price)] for price in PRICES ])
        print(f"Finished Q-Learning with estimated value {optimal_value:.2f}")
        plot_policy(policy_ql, title="Q-Learning: Learned Pricing Policy")
        plot_metrics(metrics_ql, "Q-Learning: Evolution of Metrics")

    if args.method in ["all", "sarsa"]:
        Q_sarsa, policy_sarsa, metrics_sarsa = sarsa_lambda(verbose=args.verbose)
        optimal_value = max([ Q_sarsa[(K, T, price)] for price in PRICES ])
        print(f"Finished SARSA(λ={TRACE_DECAY:.2f}) with estimated value {optimal_value:.2f}")
        plot_policy(policy_sarsa, title=f"SARSA(λ={TRACE_DECAY:.2f}): Learned Pricing Policy")
        plot_metrics(metrics_sarsa, f"SARSA(λ={TRACE_DECAY:.2f}): Evolution of Metrics")
