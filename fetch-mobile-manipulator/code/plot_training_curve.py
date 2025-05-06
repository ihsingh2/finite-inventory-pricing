import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import constants
from model import DDPG_HER, SAC_HER, TD3_HER


def plot_with_sliding_window(epochs, data, label, window_size=3):
    """Plots data with a sliding window average."""

    padding = window_size // 2
    padded_data = np.pad(data, (padding, padding), mode='edge')
    smoothed_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    plt.plot(epochs, smoothed_data, label=label, linestyle='-')


def main():
    """ Main function to generate plots. """

    # Dictionaries to store loaded data
    all_success_rates = {}
    all_rewards = {}
    max_epochs = 0

    # --- Load Data ---
    print("Loading data from checkpoints...")
    for algo in ["DDPG", "SAC", "TD3"]:
        model_filename = f"{constants.ENV_NAME}_{algo}.pt"
        model_path = os.path.join(constants.SAVE_DIR, model_filename)

        if os.path.exists(model_path):
            try:
                # Load checkpoint (use map_location='cpu' if plotting on a machine without GPU)
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                # Extract history data (handle missing keys gracefully)
                success_hist = checkpoint.get('history_success_rate', [])
                reward_hist = checkpoint.get('history_reward', [])

                if success_hist and reward_hist:
                    all_success_rates[algo] = np.array(success_hist)
                    all_rewards[algo] = np.array(reward_hist)
                    max_epochs = max(max_epochs, len(success_hist))
                    print(f"Loaded {len(success_hist)} epochs for {algo}")
                else:
                    print(f"Warning: History data missing or empty in {model_filename}")

            except Exception as e:
                print(f"Error loading {model_filename}: {e}")
        else:
            print(f"Checkpoint file not found: {model_filename}")

    # --- Plotting ---
    if not all_success_rates or not all_rewards:
        print("\nNo data loaded. Cannot create plots.")
    else:
        epochs = np.arange(1, max_epochs + 1)

        # Plot 1: Success Rate vs. Epochs
        plt.figure(figsize=(10, 6))
        for algo, rates in all_success_rates.items():
            current_epochs = np.arange(1, len(rates) + 1)
            plot_with_sliding_window(current_epochs, rates, algo)

        plt.xlabel("Epoch")
        plt.ylabel("Success Rate")
        plt.title(f"Success Rate vs Epochs ({constants.ENV_NAME})")
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"../slides/figures/{constants.ENV_NAME}_success_rate.png")
        plt.close()
        plt.clf()

        # Plot 2: Average Reward vs. Epochs
        plt.figure(figsize=(10, 6))
        for algo, rewards in all_rewards.items():
            current_epochs = np.arange(1, len(rewards) + 1)
            plot_with_sliding_window(current_epochs, rewards, algo)

        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.title(f"Average Reward vs Epochs ({constants.ENV_NAME})")
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"../slides/figures/{constants.ENV_NAME}_reward.png")
        plt.close()
        plt.clf()


if __name__ == "__main__":
    main()
