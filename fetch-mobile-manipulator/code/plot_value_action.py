import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import gymnasium as gym
import gymnasium_robotics
import constants
from model import Normaliser, ActorDet, ActorStoch, Critic, BaseAgent, DDPG_HER, SAC_HER, TD3_HER

# Configuration
TARGET_ENV_NAME = "FetchReach-v3"
ALGOS_TO_COMPARE = ["DDPG", "TD3", "SAC"]
NUM_POINTS = 100

# Determine device
device = torch.device("cuda" if constants.CUDA and torch.cuda.is_available() else "cpu")

# Representative base state (observation): [x, y, z, r_grip, l_grip, vx, vy, vz, v_rgrip, v_lgrip]
base_observation = np.array([
    1.34, 0.75, 0.53,  # approx center workspace pos x, y, z
    0.0, 0.0,          # closed gripper joints
    0.0, 0.0, 0.0,     # zero linear velocity vx, vy, vz
    0.0, 0.0           # zero gripper velocity
], dtype=np.float32)

# Desired goal
desired_goal = np.array([
    1.40, 0.75, 0.60   # Goal x, y, z
], dtype=np.float32)

# Define ranges to vary each observation and goal dimension
obs_dim_ranges = [
    (1.1, 1.6),   # Dim 0: End effector x position
    (0.5, 1.0),   # Dim 1: End effector y position
    (0.4, 0.8),   # Dim 2: End effector z position
    (-0.01, 0.05),# Dim 3: Right gripper joint displacement (0 closed, 0.05 open?)
    (-0.01, 0.05),# Dim 4: Left gripper joint displacement
    (-1.0, 1.0),  # Dim 5: End effector vx
    (-1.0, 1.0),  # Dim 6: End effector vy
    (-1.0, 1.0),  # Dim 7: End effector vz
    (-0.5, 0.5),  # Dim 8: Right gripper velocity
    (-0.5, 0.5),  # Dim 9: Left gripper velocity
]
goal_dim_ranges = [
    (1.1, 1.6),   # Goal X range (matches Obs Dim 0 range)
    (0.5, 1.0),   # Goal Y range (matches Obs Dim 1 range)
    (0.4, 0.8),   # Goal Z range (matches Obs Dim 2 range)
]

# Descriptive labels for dimensions
obs_dim_labels = [
    "EE Pos X", "EE Pos Y", "EE Pos Z",
    "R Grip Joint", "L Grip Joint",
    "EE Vel X", "EE Vel Y", "EE Vel Z",
    "R Grip Vel", "L Grip Vel"
]
goal_dim_labels = [
    "Goal Pos X", "Goal Pos Y", "Goal Pos Z"
]
action_dim_labels = ["Delta X", "Delta Y", "Delta Z"]

def load_agent_components(env_name, algo):
    """Loads the specified agent model and returns relevant components."""

    model_filename = f"{env_name}_{algo}.pt"
    model_path = os.path.join(constants.SAVE_DIR, model_filename)

    if not os.path.exists(model_path):
        print(f"Warning: Model file not found, skipping {algo}. Path: {model_path}")
        return None

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint for {algo}: {e}")
        return None

    # --- Instantiate necessary components ---
    try:
        # Use a dummy env instance to get parameters needed for network init
        dummy_env = gym.make(env_name)
        env_params = BaseAgent(dummy_env, algo, device).ep_p
        dummy_env.close()
    except Exception as e:
        print(f"Warning: Could not create dummy env '{env_name}' for {algo}. Using hardcoded params. Error: {e}")
        # Fallback hardcoded params
        env_params = {'obs': 10, 'g': 3, 'act': 4, 'act_max': 1.0}

    # Instantiate Normalizers
    o_norm = Normaliser(env_params['obs'], clip_range=constants.CLIP_RANGE)
    g_norm = Normaliser(env_params['g'], clip_range=constants.CLIP_RANGE)

    # Instantiate Networks
    actor = None
    critic = None

    try:
        if algo == "DDPG":
            actor = ActorDet(env_params['obs'], env_params['g'], env_params['act'], env_params['act_max']).to(device)
            critic = Critic(env_params['obs'], env_params['g'], env_params['act'], env_params['act_max']).to(device)
            actor.load_state_dict(checkpoint['actor_state_dict'])
            critic.load_state_dict(checkpoint['critic_state_dict'])
        elif algo == "TD3":
            actor = ActorDet(env_params['obs'], env_params['g'], env_params['act'], env_params['act_max']).to(device)
            critic = Critic(env_params['obs'], env_params['g'], env_params['act'], env_params['act_max']).to(device)
            actor.load_state_dict(checkpoint['actor_state_dict'])
            critic.load_state_dict(checkpoint['critic1_state_dict']) # Load critic 1
        elif algo == "SAC":
            actor = ActorStoch(env_params['obs'], env_params['g'], env_params['act'], env_params['act_max']).to(device)
            critic = Critic(env_params['obs'], env_params['g'], env_params['act'], env_params['act_max']).to(device)
            actor.load_state_dict(checkpoint['actor_state_dict'])
            critic.load_state_dict(checkpoint['critic1_state_dict']) # Load critic 1
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        # Load normalizer states (check if keys exist)
        if 'o_norm_state' in checkpoint and 'g_norm_state' in checkpoint:
            o_norm.set_state(checkpoint['o_norm_state'])
            g_norm.set_state(checkpoint['g_norm_state'])
        else:
            print(f"Warning: Normalizer state not found in checkpoint for {algo}.")
            # Decide how to handle this: raise error, use default, etc.
            # Using default might lead to inaccurate Q-values if data was normalized.

    except KeyError as e:
        print(f"Error loading state dict key for {algo}: {e}. Checkpoint keys: {list(checkpoint.keys())}")
        return None
    except Exception as e:
        print(f"Error instantiating or loading networks/normalizers for {algo}: {e}")
        return None

    # Set networks to evaluation mode
    actor.eval()
    critic.eval()

    return actor, critic, o_norm, g_norm


if __name__ == "__main__":

    # Output directory
    figure_save_dir = "../slides/figures"
    os.makedirs(figure_save_dir, exist_ok=True)

    # --- Load Components for ALL Algorithms First ---
    loaded_components = {}
    available_algos = []
    for algo in ALGOS_TO_COMPARE:
        components = load_agent_components(TARGET_ENV_NAME, algo)
        if components:
            loaded_components[algo] = components
            available_algos.append(algo)

    if not available_algos:
        print("No models loaded successfully. Exiting.")
        exit()

    # --- Observation Dimension Analysis ---
    for dim_idx in range(base_observation.shape[0]):
        current_label = obs_dim_labels[dim_idx]

        # Get the range and generate points
        min_val, max_val = obs_dim_ranges[dim_idx]
        dim_values = np.linspace(min_val, max_val, NUM_POINTS)

        # Dictionary to store results for this dimension, keyed by algorithm
        values_per_algo = {algo: [] for algo in available_algos}
        actions_per_algo = {algo: [] for algo in available_algos}

        # Iterate through each successfully loaded algorithm
        for algo in available_algos:
            actor, critic, o_norm, g_norm = loaded_components[algo]

            with torch.no_grad(): # Disable gradient calculations for evaluation
                for val in dim_values:
                    # Create the modified observation
                    current_obs = base_observation.copy()
                    current_obs[dim_idx] = val

                    # Normalize observation and the fixed goal using this algo's normalizers
                    obs_norm = o_norm.norm(current_obs)
                    goal_norm = g_norm.norm(desired_goal)

                    # Prepare input for actor (obs + goal)
                    pi_input = torch.tensor(
                        np.concatenate([obs_norm, goal_norm]), dtype=torch.float32, device=device
                    ).unsqueeze(0)

                    # Get action from the actor
                    if algo == "SAC":
                        action_tensor, _, _ = actor(pi_input, deterministic=True, with_logprob=False)
                    else: # DDPG, TD3
                        action_tensor = actor(pi_input)

                    # Prepare input for critic
                    obs_goal_input = pi_input

                    # Get Q-value from the critic
                    q_tensor = critic(obs_goal_input, action_tensor)

                    # Store results for this algorithm
                    values_per_algo[algo].append(q_tensor.squeeze().cpu().item())
                    actions_per_algo[algo].append(action_tensor.squeeze().cpu().numpy())

        # --- Plotting Comparison for the current observation dimension ---
        plt.figure(figsize=(10, 6))
        for algo, q_values in values_per_algo.items():
            plt.plot(dim_values, q_values, label=algo)
        plt.xlabel(f"{current_label}")
        plt.ylabel("Estimated Q-Value")
        plt.title(f"Q-Value vs {current_label}")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()

        filename_label = current_label.replace(" ", "_").replace("/", "")
        plot_filename = os.path.join(figure_save_dir, f"{TARGET_ENV_NAME}_q_value_obs_{filename_label}.png")
        plt.savefig(plot_filename)
        plt.close()
        plt.clf()

        num_action_dims = len(action_dim_labels)
        for act_idx in range(num_action_dims):
            action_label = action_dim_labels[act_idx]
            plt.figure(figsize=(10, 6))

            for algo, all_actions in actions_per_algo.items():
                actions_np = np.array(all_actions)
                action_dim_data = actions_np[:, act_idx]
                plt.plot(dim_values, action_dim_data, label=f"{algo}")

            varied_dim_label = current_label
            plt.xlabel(f"{varied_dim_label}")
            plt.ylabel(f"Predicted Action ({action_label})")
            plt.title(f"Action ({action_label}) vs {varied_dim_label}")
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()

            varied_filename_label = varied_dim_label.replace(" ", "_").replace("/", "")
            action_filename_label = action_label.replace(" ", "_").replace("/", "")
            plot_filename = os.path.join(
                figure_save_dir,
                f"{TARGET_ENV_NAME}_action_obs_{varied_filename_label}_vs_{action_filename_label}.png"
            )
            plt.savefig(plot_filename)
            plt.close()
            plt.clf()

    # --- Goal Dimension Analysis ---
    for goal_dim_idx in range(desired_goal.shape[0]):
        current_label = goal_dim_labels[goal_dim_idx]

        # Get the range and generate points for the current goal dimension
        min_val, max_val = goal_dim_ranges[goal_dim_idx]
        goal_dim_values = np.linspace(min_val, max_val, NUM_POINTS)

        # Dictionary to store results for this dimension, keyed by algorithm
        values_per_algo = {algo: [] for algo in available_algos}
        actions_per_algo = {algo: [] for algo in available_algos}

        # Iterate through each successfully loaded algorithm
        for algo in available_algos:
            actor, critic, o_norm, g_norm = loaded_components[algo]

            with torch.no_grad(): # Disable gradient calculations for evaluation
                for val in goal_dim_values:
                    # Keep the observation fixed at the base state
                    current_obs = base_observation.copy()

                    # Create the modified goal
                    current_goal = desired_goal.copy()
                    current_goal[goal_dim_idx] = val

                    # Normalize the fixed observation and the VARYING goal using this algo's normalizers
                    obs_norm = o_norm.norm(current_obs)
                    goal_norm = g_norm.norm(current_goal)

                    # Prepare input for actor (obs + goal)
                    pi_input = torch.tensor(
                        np.concatenate([obs_norm, goal_norm]), dtype=torch.float32, device=device
                    ).unsqueeze(0)

                    # Get action from the actor for this state-goal pair
                    if algo == "SAC":
                        action_tensor, _, _ = actor(pi_input, deterministic=True, with_logprob=False)
                    else: # DDPG, TD3
                        action_tensor = actor(pi_input)

                    # Prepare input for critic
                    obs_goal_input = pi_input

                    # Get Q-value from the critic
                    q_tensor = critic(obs_goal_input, action_tensor)

                    # Store results for this algorithm
                    values_per_algo[algo].append(q_tensor.squeeze().cpu().item())
                    actions_per_algo[algo].append(action_tensor.squeeze().cpu().numpy())

        # --- Plotting Comparison for the current goal dimension ---
        plt.figure(figsize=(10, 6))
        for algo, q_values in values_per_algo.items():
            plt.plot(goal_dim_values, q_values, label=algo)
        plt.xlabel(f"{current_label}")
        plt.ylabel("Estimated Q-Value")
        plt.title(f"Q-Value vs {current_label}")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()

        filename_label = current_label.replace(" ", "_").replace("/", "")
        plot_filename = os.path.join(figure_save_dir, f"{TARGET_ENV_NAME}_q_value_goal_{filename_label}.png")
        plt.savefig(plot_filename)
        plt.close()
        plt.clf()

        num_action_dims = len(action_dim_labels)
        for act_idx in range(num_action_dims):
            action_label = action_dim_labels[act_idx]
            plt.figure(figsize=(10, 6))

            for algo, all_actions in actions_per_algo.items():
                actions_np = np.array(all_actions)
                action_dim_data = actions_np[:, act_idx]
                plt.plot(goal_dim_values, action_dim_data, label=f"{algo}")

            varied_dim_label = current_label
            plt.xlabel(f"{varied_dim_label}")
            plt.ylabel(f"Predicted Action ({action_label})")
            plt.title(f"Action ({action_label}) vs {varied_dim_label}")
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()

            varied_filename_label = varied_dim_label.replace(" ", "_").replace("/", "")
            action_filename_label = action_label.replace(" ", "_").replace("/", "")
            plot_filename = os.path.join(
                figure_save_dir,
                f"{TARGET_ENV_NAME}_action_goal_{varied_filename_label}_vs_{action_filename_label}.png"
            )
            plt.savefig(plot_filename)
            plt.close()
            plt.clf()
