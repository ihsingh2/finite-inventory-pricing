import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import constants
from model import DDPG_HER, SAC_HER, TD3_HER


def main():
    """ Main function to load and run a saved model. """

    # Output directory
    figure_save_dir = "../slides/figures"
    os.makedirs(figure_save_dir, exist_ok=True)

    # Determine the computation device (GPU if available and requested, otherwise CPU)
    device = (torch.device("cuda") if constants.CUDA and torch.cuda.is_available()
            else torch.device("cpu"))

    # Create an environment instance
    eval_env = gym.make(constants.ENV_NAME)

    # Models to iterate over
    algos = ["DDPG", "SAC", "TD3"]

    # Margin of error for each model
    error_margin = {load_algo: [] for load_algo in algos}

    # Iterate over all models
    for load_algo in algos:

        model_file = os.path.join(constants.SAVE_DIR, f"{constants.ENV_NAME}_{load_algo}.pt")

        # Instantiate the corresponding agent class
        if load_algo == "SAC":
            eval_agent = SAC_HER(eval_env, device)
        elif load_algo == "TD3":
            eval_agent = TD3_HER(eval_env, device)
        elif load_algo == "DDPG":
            eval_agent = DDPG_HER(eval_env, device)
        else:
            print(f"Error: Unknown algorithm '{load_algo}' for loading.")
            exit()

        # Load the saved model weights and normalizer states
        if eval_agent.load_model():
            # Set networks to evaluation mode (disables dropout, etc.)
            eval_agent.act.eval()
            if hasattr(eval_agent, '_critic1_net'): eval_agent._critic1_net.eval()
            if hasattr(eval_agent, '_critic2_net'): eval_agent._critic2_net.eval()
            if hasattr(eval_agent, '_critic_net'): eval_agent._critic_net.eval()
        else:
            print("Failed to load the model.")

        # Find the margin of error
        for i in range(500):
            o, _ = eval_env.reset()
            obs = o['observation'].astype(np.float32)
            g_a = o['achieved_goal'].astype(np.float32)
            g = o['desired_goal'].astype(np.float32)
            done = False
            ep_success = False

            # Run one evaluation episode
            while not done:
                # Get deterministic action from policy
                with torch.no_grad():
                    pi_input = eval_agent._pi_in(obs, g)
                    # Get deterministic action (mean for SAC)
                    if eval_agent.algo == "SAC":
                        action, _, _ = eval_agent.act(pi_input, deterministic=True, with_logprob=False)
                    else: # DDPG, TD3
                        action = eval_agent.act(pi_input)

                    action = action.cpu().numpy().squeeze()

                # Step environment
                o2, reward, terminated, truncated, info = eval_env.step(action)

                done = terminated or truncated
                obs = o2['observation'].astype(np.float32)
                g_a = o2['achieved_goal'].astype(np.float32)
                g = o2['desired_goal'].astype(np.float32)

                # Check for success at the end of the episode
                if done:
                    # Fetch envs store success in info['is_success']
                    ep_success = info.get('is_success', False)

            error_margin[load_algo].append(np.linalg.norm(g_a - g))

    # Plot the graph
    plt.figure(figsize=(10, 6))
    for load_algo in algos:
        plt.hist(error_margin[load_algo], label=load_algo, bins=50, alpha=0.5)
    plt.axvline(x=0.05, color='red', linestyle='--', label='Acceptable Threshold')
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Distance between the Desired Goal and Achieved Goal ({constants.ENV_NAME})")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_save_dir, f"{constants.ENV_NAME}_error_margin.png"))
    plt.close()
    plt.clf()

    # Close the evaluation environment
    eval_env.close()


if __name__ == "__main__":
    main()
