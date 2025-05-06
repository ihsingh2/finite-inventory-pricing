# Common parameters
ENV_NAME     = "FetchReach-v3"    # Environment ID (e.g., FetchReach-v3, FetchPush-v3)
N_EPOCHS     = 25                 # Number of training epochs
N_CYCLES     = 50                 # Number of cycles per epoch (data collection + training)
N_BATCHES    = 40                 # Number of gradient steps per cycle
N_TEST       = 25                 # Number of episodes for evaluation per epoch
ROLLOUTS     = 2                  # Number of rollouts (episodes) collected per cycle
CLIP_RANGE   = 5.                 # Clipping standard deviation for observations

# Exploration strategy parameters
NOISE_EPS    = 0.2                # Std deviation of Gaussian noise added to actions for exploration
RANDOM_EPS   = 0.3                # Probability of taking a completely random action

# Optimisation parameters
LR_ACTOR     = 5e-4               # Learning rate for the actor network
LR_CRITIC    = 5e-4               # Learning rate for the critic network(s)
GAMMA        = 0.98               # Discount factor for future rewards
POLYAK       = 0.95               # Coefficient for Polyak-averaging (soft target network updates)

# Hindsight Experience Replay (HER) parameters
REPLAY_K     = 4                  # Ratio of HER goals to original goals (k future strategy)
BUFFER_SZ    = int(1e6)           # Maximum size of the replay buffer (number of transitions)
BATCH_SZ     = 256                # Batch size for sampling from the replay buffer

# TD3 specific parameters
POLICY_DELAY = 2                  # Frequency of delayed policy updates
TARGET_NOISE = 0.2                # Std deviation of noise added to target policy smoothing
NOISE_CLIP   = 0.5                # Clipping range for target policy smoothing noise

# SAC specific parameters
LR_ALPHA       = 5e-4             # Learning rate for the entropy temperature (alpha)
ALPHA_INIT     = 0.2              # Initial value for alpha (temperature parameter)
TARGET_ENTROPY = "auto"           # Target entropy for automatic temperature tuning (-|A| if "auto")
LOG_STD_MIN    = -20              # Clamp log std deviations for numerical stability
LOG_STD_MAX    = 2                # Clamp log std deviations for numerical stability

# Miscellaneous parameters
SEED         = 123                # Random seed for reproducibility
CUDA         = True               # Whether to use GPU (if available)
SAVE_DIR     = "models"           # Directory to save trained models
