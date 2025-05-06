import gymnasium as gym
import gymnasium_robotics
import os
import torch
import constants
from model import DDPG_HER, SAC_HER, TD3_HER


def main():
    """ Main function to load and run a saved model. """

    # Determine the computation device (GPU if available and requested, otherwise CPU)
    device = (torch.device("cuda") if constants.CUDA and torch.cuda.is_available()
            else torch.device("cpu"))

    # Choose the algorithm (e.g., "DDPG", "SAC", "TD3") and environment name of the model to load
    load_algo = "SAC"
    load_env_name = constants.ENV_NAME
    model_file = os.path.join(constants.SAVE_DIR, f"{load_env_name}_{load_algo}.pt")
    print(f"\n=== Loading and evaluating {load_algo} on {load_env_name} ===")

    # Check if the model file exists
    if os.path.exists(model_file):

        # Create a new environment instance, this time with rendering enabled
        print("Creating environment with rendering...")
        eval_env = gym.make(load_env_name, render_mode="human")

        # Instantiate the corresponding agent class
        if load_algo == "SAC":
            eval_agent = SAC_HER(eval_env, device)
        elif load_algo == "TD3":
            eval_agent = TD3_HER(eval_env, device)
        elif load_algo == "DDPG":
            eval_agent = DDPG_HER(eval_env, device)
        else:
            print(f"Error: Unknown algorithm '{load_algo}' for loading.")
            eval_agent = None

        if eval_agent:
            # Load the saved model weights and normalizer states
            if eval_agent.load_model():
                # Set networks to evaluation mode (disables dropout, etc.)
                eval_agent.act.eval()
                if hasattr(eval_agent, '_critic1_net'): eval_agent._critic1_net.eval()
                if hasattr(eval_agent, '_critic2_net'): eval_agent._critic2_net.eval()
                if hasattr(eval_agent, '_critic_net'): eval_agent._critic_net.eval()

                # Run evaluation with rendering
                print("Running evaluation with rendering...")
                final_success_rate, final_reward = eval_agent.evaluate(render=True)
                print(f"\nEvaluation complete. Success rate: {final_success_rate:.3f}. Reward: {final_reward:.3f}")
            else:
                print("Failed to load the model.")

        # Close the evaluation environment
        eval_env.close()
        print("Evaluation environment closed.")
    else:
        print(f"Model file not found: {model_file}")


if __name__ == "__main__":
    main()
