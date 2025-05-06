import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pathlib
import random
import torch
import constants
from model import DDPG_HER, SAC_HER, TD3_HER


def main():
    """ Main function to create environment, agent, and start training. """

    # Determine the computation device (GPU if available and requested, otherwise CPU)
    device = (torch.device("cuda") if constants.CUDA and torch.cuda.is_available()
            else torch.device("cpu"))

    # Set random seeds for reproducibility across NumPy, Python's random, and PyTorch
    np.random.seed(constants.SEED)
    random.seed(constants.SEED)
    torch.manual_seed(constants.SEED)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed_all(constants.SEED)

    # Create the directory for saving models
    pathlib.Path(constants.SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # Create the Gym environment
    env = gym.make(constants.ENV_NAME)

    # Define the algorithms and their corresponding agent classes to train
    agents_to_train = {
        "DDPG": DDPG_HER,
        "SAC": SAC_HER,
        "TD3": TD3_HER
    }

    # Loop through each algorithm, create an agent, train it, and save the model
    for algo_name, agent_class in agents_to_train.items():
        print(f"\n=== Training {algo_name} on {constants.ENV_NAME} ===")
        agent = agent_class(env, device)
        agent.train()
        agent.save_model()
        print(f"=== Finished training {algo_name} ===")

    # Close the environment after training is complete
    env.close()
    print("\nAll training finished.")


if __name__ == "__main__":
    main()
