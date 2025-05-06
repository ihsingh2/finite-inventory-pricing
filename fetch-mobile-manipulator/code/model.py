import os
import time
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import constants


def init_(m):
    """ Initialises weights of a linear layer using Kaiming Uniform. """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        # Optionally initialize biases (often zeros)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Normaliser:
    """ Normalises input data using running mean and standard deviation. """
    def __init__(self, size, eps=1e-2, clip_range=np.inf):
        """
        Initialises the normaliser.
        Args:
            size (int): The dimension of the data to normalise.
            eps (float): A small epsilon added to the variance to prevent division by zero.
            clip_range (float): The range to clip the normalised data to [-clip_range, clip_range].
        """
        self.size = size
        self.eps = eps
        self.clip_range = clip_range

        # Local buffers for sums and counts within a cycle
        self.lsum = np.zeros(size, np.float32)
        self.lsumsq = np.zeros(size, np.float32)
        self.lcnt = 0

        # Total sums and counts accumulated across cycles
        self.tsum = np.zeros(size, np.float32)
        self.tsumsq = np.zeros(size, np.float32)
        self.tcnt = 1e-4 # Start with a small count to avoid division by zero initially

        # Running mean and standard deviation
        self.mean = np.zeros(size, np.float32)
        self.std = np.ones(size, np.float32)

    def update(self, v):
        """ Adds a batch of data to the local buffer for statistics calculation. """
        v = v.reshape(-1, self.size).astype(np.float32) # Ensure correct shape and type
        self.lsum += v.sum(axis=0)
        self.lsumsq += (v**2).sum(axis=0)
        self.lcnt += v.shape[0]

    def recompute(self):
        """ Updates the running mean and std using data from the local buffer and resets the buffer. """
        if self.lcnt == 0: # Skip if no new data was added
            return

        # Update total stats with local stats
        self.tsum += self.lsum
        self.tsumsq += self.lsumsq
        self.tcnt += self.lcnt

        # Reset local stats
        self.lsum.fill(0)
        self.lsumsq.fill(0)
        self.lcnt = 0

        # Recalculate mean and std
        self.mean = self.tsum / self.tcnt
        variance = self.tsumsq / self.tcnt - self.mean**2
        # Ensure variance is non-negative and add epsilon
        self.std = np.sqrt(np.maximum(variance, self.eps))

    def norm(self, v):
        """ Normalises input data v using the running mean and std, and clips the result. """
        return np.clip((v - self.mean) / self.std, -self.clip_range, self.clip_range)

    # Add methods to get and set state for saving/loading
    def get_state(self):
        return {
            'tsum': self.tsum,
            'tsumsq': self.tsumsq,
            'tcnt': self.tcnt,
            'mean': self.mean,
            'std': self.std
        }

    def set_state(self, state):
        self.tsum = state['tsum']
        self.tsumsq = state['tsumsq']
        self.tcnt = state['tcnt']
        self.mean = state['mean']
        self.std = state['std']
        # Reset local buffers as they are transient
        self.lsum.fill(0)
        self.lsumsq.fill(0)
        self.lcnt = 0


class HER:
    """ Implements Hindsight Experience Replay (HER) using the 'future' strategy. """
    def __init__(self, k, compute_reward):
        """
        Initialises the HER sampler.
        Args:
            k (int): The ratio of HER goals to original goals (defines future_p).
            compute_reward (callable): The environment's reward function.
        """
        self.future_p = 1 - (1. / (1 + k)) # Probability of sampling a future goal
        self.compute_reward = compute_reward # Function to compute reward based on (ag, g, info)
        self.rng = np.random.default_rng() # Local random number generator

    def sample(self, batch, batch_size):
        """
        Samples transitions from a batch and potentially replaces goals with achieved goals
        from the future in the same trajectory (HER).
        Args:
            batch (dict): A dictionary containing episode data (obs, ag, g, actions).
            batch_size (int): The number of transitions to sample.
        Returns:
            dict: A dictionary containing the sampled (and potentially HER-modified) transitions.
        """
        T = batch['actions'].shape[1] # Timesteps per episode in the batch
        B = batch['actions'].shape[0] # Number of episodes in the batch

        # Select random episodes and timesteps
        ep_indices = self.rng.integers(0, B, batch_size)
        t_indices = self.rng.integers(0, T, batch_size)

        # Get the original transitions for the selected indices
        transitions = {k: batch[k][ep_indices, t_indices].copy() for k in batch}

        # Determine which transitions will use HER (future strategy)
        her_mask = self.rng.random(batch_size) < self.future_p

        # Calculate future offsets only for transitions using HER
        # Ensure future offset is at least 1 and within the episode boundary
        future_offset = self.rng.integers(0, T - t_indices)
        future_t = t_indices + future_offset

        # Replace the 'g' (goal) with a future 'ag' (achieved_goal) for HER transitions
        transitions['g'][her_mask] = batch['ag'][ep_indices[her_mask], future_t[her_mask]]

        # Recompute the reward 'r' based on the (potentially modified) goal
        # The reward is computed based on the next state's achieved goal and the new goal
        transitions['r'] = self.compute_reward(
            transitions['ag_next'], transitions['g'], {} # Pass empty info dict
        ).astype(np.float32).reshape(-1, 1) # Ensure correct shape and type

        return transitions


class ReplayBuffer:
    """ Stores transitions and allows sampling with HER. """
    def __init__(self, env_params, size, k, reward_fn):
        """
        Initialises the replay buffer.
        Args:
            env_params (dict): Dictionary containing environment parameters (dims, max_T).
            size (int): Maximum number of transitions to store.
            k (int): The HER parameter (ratio of HER goals).
            reward_fn (callable): The environment's reward function for HER.
        """
        self.T = env_params['max_T'] # Max episode length
        self.size = size // self.T # Buffer size in terms of number of episodes
        self.ptr = 0 # Current insertion point (index)
        self.full = False # Flag indicating if the buffer has wrapped around

        # Pre-allocate buffer memory for different transition components
        # +1 for obs and ag to store the final state/achieved_goal of the episode
        self.buf = {
            'obs': np.empty([self.size, self.T + 1, env_params['obs']], np.float32),
            'ag': np.empty([self.size, self.T + 1, env_params['ag']], np.float32),
            'g': np.empty([self.size, self.T, env_params['g']], np.float32), # Goal is constant within an episode
            'actions': np.empty([self.size, self.T, env_params['act']], np.float32)
        }

        # Initialise the HER sampler
        self.her = HER(k, reward_fn)
        self.rng = np.random.default_rng() # Local random number generator

    def store(self, episode_batch):
        """ Stores a batch of episodes in the buffer. """
        batch_size = episode_batch['actions'].shape[0] # Number of episodes in the batch
        # Calculate indices for insertion, handling wrap-around
        indices = np.arange(self.ptr, self.ptr + batch_size) % self.size

        # Store data for each key
        for key in self.buf:
            self.buf[key][indices] = episode_batch[key]

        # Update pointer and full flag
        self.ptr = (self.ptr + batch_size) % self.size
        self.full = self.full or (self.ptr == 0 and batch_size > 0) # Buffer becomes full if pointer wraps to 0

    def sample(self, batch_size):
        """ Samples a batch of transitions, applying HER. """
        # Determine the maximum index to sample from (current pointer or full size)
        max_index = self.size if self.full else self.ptr
        if max_index == 0: # Cannot sample if buffer is empty
            return None

        # Sample random episode indices
        ep_indices = self.rng.integers(0, max_index, batch_size)

        # Retrieve the full episodes corresponding to the sampled indices
        episode_batch = {k: self.buf[k][ep_indices] for k in self.buf}

        # Prepare batch for HER by adding next obs and ag
        episode_batch['obs_next'] = episode_batch['obs'][:, 1:, :]
        episode_batch['ag_next'] = episode_batch['ag'][:, 1:, :]
        # Note: episode_batch['obs'] and ['ag'] still contain the T+1 length arrays

        # Use HER sampler to get the final batch of transitions
        transitions = self.her.sample(episode_batch, batch_size)
        return transitions


class ActorDet(nn.Module):
    """ Deterministic Actor Network (used in DDPG/TD3). """
    def __init__(self, obs_sz, goal_sz, act_sz, act_max):
        """
        Initialises the deterministic actor.
        Args:
            obs_sz (int): Dimension of the observation space.
            goal_sz (int): Dimension of the goal space.
            act_sz (int): Dimension of the action space.
            act_max (float): Maximum absolute value of the actions.
        """
        super().__init__()
        self.max = act_max # Store max action value for scaling output
        # Define the network architecture
        self.net = nn.Sequential(
            nn.Linear(obs_sz + goal_sz, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_sz), nn.Tanh() # Tanh activation scales output to [-1, 1]
        )
        # Apply custom weight initialisation
        self.apply(init_)

    def forward(self, x):
        """ Forward pass: maps observation and goal to a deterministic action. """
        # Scale the Tanh output to the environment's action range [-act_max, act_max]
        return self.max * self.net(x)


class ActorStoch(nn.Module):
    """ Stochastic Actor Network (Gaussian policy, used in SAC). """
    def __init__(self, obs_sz, goal_sz, act_sz, act_max):
        """
        Initialises the stochastic actor.
        Args:
            obs_sz (int): Dimension of the observation space.
            goal_sz (int): Dimension of the goal space.
            act_sz (int): Dimension of the action space.
            act_max (float): Maximum absolute value of the actions (scales the output).
        """
        super().__init__()
        self.max = act_max # Store max action value for scaling output
        # Shared layers
        self.fc = nn.Sequential(
            nn.Linear(obs_sz + goal_sz, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        # Heads for mean and log standard deviation
        self.mu_head = nn.Linear(256, act_sz)
        self.logstd_head = nn.Linear(256, act_sz)
        # Apply custom weight initialisation
        self.apply(init_)

    def forward(self, x, deterministic=False, with_logprob=True):
        """
        Forward pass: maps observation and goal to a distribution over actions.
        Args:
            x (Tensor): Input tensor (concatenated observation and goal).
            deterministic (bool): If True, return the mean action (for evaluation).
            with_logprob (bool): If True, calculate and return the log probability of the action.
        Returns:
            tuple: (action, log_prob, mean_action_before_squashing)
                   log_prob is None if with_logprob=False or deterministic=True.
        """
        # Pass input through shared layers
        h = self.fc(x)
        # Get mean and log standard deviation from the heads
        mu = self.mu_head(h)
        log_std = torch.clamp(self.logstd_head(h), constants.LOG_STD_MIN, constants.LOG_STD_MAX)
        std = log_std.exp()

        logp = None # Initialise log probability
        if deterministic:
            # Use the mean action directly for deterministic evaluation
            pi = mu
        else:
            # Sample action from the Gaussian distribution: pi = mu + std * N(0, I)
            eps = torch.randn_like(mu)
            pi = mu + std * eps

        # Apply Tanh squashing to bound the action within [-1, 1] (before scaling)
        tanh_pi = torch.tanh(pi)

        if with_logprob and not deterministic:
            # Calculate log probability of the sampled action 'pi' under the Gaussian distribution
            # Formula: log P(u|s) = Sum_i [-0.5 * ((u_i - mu_i)/std_i)^2 - log(std_i) - 0.5*log(2*pi)]
            logp_gaussian = (-0.5 * ((pi - mu) / std).pow(2) - log_std - 0.5 * np.log(2 * np.pi)).sum(dim=1, keepdim=True)

            # Apply correction for the Tanh squashing transformation
            # Formula: log P(a|s) = log P(u|s) - Sum_i log(1 - tanh(u_i)^2)
            # Stable version: log P(a|s) = log P(u|s) - Sum_i [2 * (log(2) - u_i - softplus(-2*u_i))]
            logp = logp_gaussian - (2 * (np.log(2) - pi - F.softplus(-2 * pi))).sum(dim=1, keepdim=True)

        # Scale the Tanh output to the environment's action range [-act_max, act_max]
        action = self.max * tanh_pi
        # Also return the mean action scaled by Tanh and act_max (useful for some logging/debugging)
        mean_action = self.max * torch.tanh(mu)

        return action, logp, mean_action


class Critic(nn.Module):
    """ Critic Network Q(s, a) (used in DDPG/TD3/SAC). """
    def __init__(self, obs_sz, goal_sz, act_sz, act_max):
        """
        Initialises the critic.
        Args:
            obs_sz (int): Dimension of the observation space.
            goal_sz (int): Dimension of the goal space.
            act_sz (int): Dimension of the action space.
            act_max (float): Maximum absolute value of the actions (used for scaling input actions).
        """
        super().__init__()
        self.max = act_max # Store max action value for scaling input action
        # Define the network architecture
        self.net = nn.Sequential(
            # Input includes observation, goal, and action
            nn.Linear(obs_sz + goal_sz + act_sz, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1) # Outputs a single Q-value
        )
        # Apply custom weight initialisation
        self.apply(init_)

    def forward(self, x, a):
        """ Forward pass: maps observation, goal, and action to a Q-value. """
        # Scale the input action 'a' to be roughly in the range [-1, 1] before concatenating
        # This assumes the network performs better with normalised inputs
        return self.net(torch.cat([x, a / self.max], dim=1))


class BaseAgent:
    """ Base class for RL agents, handling interaction with the environment,
        data collection, normalisation, and evaluation. """
    def __init__(self, env, algo, device):
        """
        Initialises the base agent.
        Args:
            env (gym.Env): The Gym environment instance.
            algo (str): The name of the algorithm (e.g., "DDPG", "TD3", "SAC").
        """
        self.env = env
        self.algo = algo
        self.device = device

        # Get environment parameters (dimensions, action range, max steps)
        self.ep_p = self._env_params()

        # Initialise normalisers for observations and goals
        self.o_norm = Normaliser(self.ep_p['obs'], clip_range=constants.CLIP_RANGE)
        self.g_norm = Normaliser(self.ep_p['g'], clip_range=constants.CLIP_RANGE)

        # Initialise the replay buffer with HER support
        self.buf = ReplayBuffer(self.ep_p, constants.BUFFER_SZ, constants.REPLAY_K,
                                env.unwrapped.compute_reward) # Use env's reward function

        # Local random number generator
        self.rng = np.random.default_rng(constants.SEED)

        # Define path for saving/loading models
        self.model_filename = f"{constants.ENV_NAME}_{self.algo}.pt"
        self.model_path = os.path.join(constants.SAVE_DIR, self.model_filename)

        # Set SAC target entropy if needed
        if self.algo == "SAC" and constants.TARGET_ENTROPY == "auto":
            constants.TARGET_ENTROPY = -float(self.ep_p['act'])

    def _env_params(self):
        """ Extracts relevant parameters from the environment. """
        # Reset env to get observation spec
        o, _ = self.env.reset(seed=constants.SEED)
        # Ensure observation is a dictionary
        if not isinstance(o, dict) or not all(k in o for k in ['observation', 'desired_goal', 'achieved_goal']):
            raise TypeError(f"Environment {constants.ENV_NAME} does not return the expected dictionary observation format.")

        params = dict(
            obs=o['observation'].shape[0],          # Observation dimension
            g=o['desired_goal'].shape[0],           # Goal dimension
            ag=o['achieved_goal'].shape[0],         # Achieved goal dimension
            act=self.env.action_space.shape[0],     # Action dimension
            act_max=float(self.env.action_space.high[0]), # Max action value (assuming symmetric)
            max_T=self.env._max_episode_steps       # Max steps per episode
        )
        # Check if action space is symmetric
        if not np.allclose(self.env.action_space.high, -self.env.action_space.low):
            warnings.warn("Action space is not symmetric. Using high value for act_max.")
        return params

    def _pi_in(self, o, g):
        """ Prepares normalised observation and goal for policy input. """
        # Normalise observation and goal
        o_norm = self.o_norm.norm(o)
        g_norm = self.g_norm.norm(g)
        # Concatenate and convert to a PyTorch tensor on the correct device
        input_tensor = torch.as_tensor(
            np.concatenate([o_norm, g_norm]), dtype=torch.float32, device=self.device
        )
        # Add batch dimension if necessary (policy expects batch input)
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)
        return input_tensor

    def _random_act(self):
        """ Samples a random action from the action space. """
        return self.rng.uniform(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            size=self.ep_p['act']
        ).astype(np.float32)

    def _explore(self, pi_out):
        """ Adds exploration noise to the deterministic policy output (for DDPG/TD3). """
        # Get the action tensor from policy output (handle potential tuples from SAC)
        action_tensor = pi_out[0] if isinstance(pi_out, tuple) else pi_out
        # Convert action to NumPy array
        action = action_tensor.cpu().numpy().squeeze()

        # Add Gaussian noise for exploration
        noise = constants.NOISE_EPS * self.ep_p['act_max'] * self.rng.standard_normal(action.shape)
        action += noise

        # Clip action to ensure it's within valid bounds
        action = np.clip(action, -self.ep_p['act_max'], self.ep_p['act_max'])

        # Epsilon-greedy exploration: with probability random_eps, take a random action
        if self.rng.random() < constants.RANDOM_EPS:
            action = self._random_act()

        return action.astype(np.float32)

    def collect_rollouts(self):
        """ Collects a specified number of rollouts (episodes) using the current policy. """
        # Dictionary to store episode data
        episode_batch = {'obs': [], 'ag': [], 'g': [], 'actions': []}

        for _ in range(constants.ROLLOUTS):
            # Reset environment and get initial state
            o, _ = self.env.reset()
            obs = o['observation'].astype(np.float32)
            ag = o['achieved_goal'].astype(np.float32)
            g = o['desired_goal'].astype(np.float32)

            # Buffers for the current episode's trajectory
            buf_obs, buf_ag, buf_g, buf_act = [], [], [], []

            # Run one episode
            for t in range(self.ep_p['max_T']):
                # Get action from policy (inference mode)
                with torch.no_grad():
                    pi_input = self._pi_in(obs, g)
                    pi_out = self.act(pi_input) # self.act must be defined in subclass

                # Apply exploration strategy based on algorithm
                if self.algo in ('DDPG', 'TD3'):
                    action = self._explore(pi_out)
                elif self.algo == 'SAC':
                    # For SAC, the policy outputs the sampled action directly
                    action_tensor = pi_out[0] if isinstance(pi_out, tuple) else pi_out
                    action = action_tensor.cpu().numpy().squeeze()
                else:
                    raise NotImplementedError(f"Exploration for algorithm {self.algo} not defined.")

                # Store current state components and action
                buf_obs.append(obs.copy())
                buf_ag.append(ag.copy())
                buf_g.append(g.copy()) # Goal is constant, but store per step for consistency
                buf_act.append(action.copy())

                # Step the environment
                o2, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated # Check if episode ended

                # Update current state for the next iteration
                obs = o2['observation'].astype(np.float32)
                ag = o2['achieved_goal'].astype(np.float32)
                # Goal might change in some envs, though usually static in Fetch
                g = o2['desired_goal'].astype(np.float32)

                if done: # Break if episode finished early
                    break

            # Append final observation and achieved goal for buffer consistency (T+1)
            buf_obs.append(obs.copy())
            buf_ag.append(ag.copy())

            # Add the collected episode trajectory to the batch
            episode_batch['obs'].append(np.array(buf_obs, np.float32))
            episode_batch['ag'].append(np.array(buf_ag, np.float32))
            episode_batch['g'].append(np.array(buf_g, np.float32))
            episode_batch['actions'].append(np.array(buf_act, np.float32))

        # Convert lists of episodes to NumPy arrays
        episode_batch = {k: np.array(v, np.float32) for k, v in episode_batch.items()}

        # Store the collected batch in the replay buffer
        self.buf.store(episode_batch)
        # Update normalisation statistics with the collected data
        self._update_norm(episode_batch)

    def _update_norm(self, ep_batch):
        """ Updates the observation and goal normalisers using data from the collected episodes. """
        # Extract relevant data from the episode batch
        mb_obs = ep_batch['obs']     # Shape: (N_rollouts, T+1, Obs_dim)
        mb_ag = ep_batch['ag']      # Shape: (N_rollouts, T+1, Ag_dim)
        mb_g = ep_batch['g']       # Shape: (N_rollouts, T, G_dim)
        mb_actions = ep_batch['actions'] # Shape: (N_rollouts, T, Act_dim)

        # Create a structure similar to buffer output for HER sampling
        # We need 'obs_next' and 'ag_next' for HER's reward calculation if needed
        # Note: shape consistency for HER sampler is important
        tmp_batch = dict(
            obs=mb_obs[:, :-1, :],       # (N, T, Obs)
            ag=mb_ag[:, :-1, :],        # (N, T, Ag)
            g=mb_g,                     # (N, T, G)
            actions=mb_actions,         # (N, T, Act)
            obs_next=mb_obs[:, 1:, :],  # (N, T, Obs)
            ag_next=mb_ag[:, 1:, :]   # (N, T, Ag)
        )

        # Sample transitions using HER logic (even if just for normalization stats)
        # This ensures normalisation stats reflect the distribution seen during training
        num_transitions_to_sample = constants.BATCH_SZ # Sample a typical batch size for stats
        if tmp_batch['actions'].shape[0] * tmp_batch['actions'].shape[1] > 0: # Check if data exists
             norm_sample = self.buf.her.sample(tmp_batch, num_transitions_to_sample)

             # Update normalisers with sampled observations, next observations, and goals
             self.o_norm.update(norm_sample['obs'])
             self.o_norm.update(norm_sample['obs_next'])
             self.g_norm.update(norm_sample['g'])

             # Recompute the running mean and standard deviation
             self.o_norm.recompute()
             self.g_norm.recompute()

    def evaluate(self, render=False):
        """ Evaluates the deterministic policy over several episodes. """
        success_list = []
        rewards_list = []
        eval_env = self.env # Use the main env or a copy if needed

        for i in range(constants.N_TEST):
            o, _ = eval_env.reset()
            obs = o['observation'].astype(np.float32)
            g = o['desired_goal'].astype(np.float32)
            done = False
            ep_success = False
            episode_reward = 0.0

            # Run one evaluation episode
            while not done:
                # Get deterministic action from policy
                with torch.no_grad():
                    pi_input = self._pi_in(obs, g)
                    # Get deterministic action (mean for SAC)
                    if self.algo == "SAC":
                        action, _, _ = self.act(pi_input, deterministic=True, with_logprob=False)
                    else: # DDPG, TD3
                        action = self.act(pi_input)

                    action = action.cpu().numpy().squeeze()

                # Step environment
                o2, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward

                if render and i == 0: # Render only the first evaluation episode if requested
                     try:
                         eval_env.render()
                         time.sleep(0.02) # Small delay for visualization
                     except Exception as e:
                         print(f"Rendering failed: {e}")
                         render = False # Disable rendering if it fails

                done = terminated or truncated
                obs = o2['observation'].astype(np.float32)
                g = o2['desired_goal'].astype(np.float32)

                # Check for success at the end of the episode
                if done:
                    # Fetch envs store success in info['is_success']
                    ep_success = info.get('is_success', False)

            success_list.append(ep_success)
            rewards_list.append(episode_reward)

        # Calculate success rate and average reward
        success_rate = float(np.mean(success_list))
        avg_reward = float(np.mean(rewards_list))

        # Return both metrics
        return success_rate, avg_reward

    def save_model(self):
        """ Saves the agent's models and normalisers. Should be implemented by subclasses. """
        raise NotImplementedError

    def load_model(self):
        """ Loads the agent's models and normalisers. Should be implemented by subclasses. """
        raise NotImplementedError

    def train(self):
        """ Main training loop. Should be implemented by subclasses. """
        raise NotImplementedError

    # Need access to the actor network for evaluation/rollouts
    @property
    def act(self):
        """ Returns the actor network. Must be implemented by subclasses. """
        raise NotImplementedError


class DDPG_HER(BaseAgent):
    """ Deep Deterministic Policy Gradient (DDPG) agent with HER. """
    def __init__(self, env, device):
        """ Initialises the DDPG agent, networks, and optimizers. """
        super().__init__(env, "DDPG", device) # Call base class constructor
        self.device = device
        p = self.ep_p # Environment parameters

        # Create actor and critic networks
        self._actor_net = ActorDet(p['obs'], p['g'], p['act'], p['act_max']).to(self.device)
        self._critic_net = Critic(p['obs'], p['g'], p['act'], p['act_max']).to(self.device)

        # Create target networks (initially identical to main networks)
        self._target_actor_net = copy.deepcopy(self._actor_net)
        self._target_critic_net = copy.deepcopy(self._critic_net)
        # Freeze target networks (gradients are not computed for them)
        for param in self._target_actor_net.parameters(): param.requires_grad = False
        for param in self._target_critic_net.parameters(): param.requires_grad = False

        # Create optimizers
        self.opt_a = torch.optim.Adam(self._actor_net.parameters(), lr=constants.LR_ACTOR)
        self.opt_c = torch.optim.Adam(self._critic_net.parameters(), lr=constants.LR_CRITIC)

        # Add history storage
        self.history_success_rate = []
        self.history_reward = []

    @property
    def act(self):
        """ Returns the actor network instance. """
        return self._actor_net

    def gradient_step(self):
        """ Performs a single gradient update step for DDPG. """
        # Sample a batch of transitions from the replay buffer
        transitions = self.buf.sample(constants.BATCH_SZ)
        if transitions is None: return # Skip if buffer is empty

        # Unpack transitions and normalise observations/goals
        o, o2, g, a, r = (transitions[k] for k in ['obs', 'obs_next', 'g', 'actions', 'r'])
        o_norm = self.o_norm.norm(o)
        o2_norm = self.o_norm.norm(o2)
        g_norm = self.g_norm.norm(g)

        # Convert numpy arrays to PyTorch tensors
        obs_norm = torch.as_tensor(np.concatenate([o_norm, g_norm], axis=1), dtype=torch.float32, device=self.device)
        obs2_norm = torch.as_tensor(np.concatenate([o2_norm, g_norm], axis=1), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(r, dtype=torch.float32, device=self.device)

        # --- Critic Update ---
        with torch.no_grad():
            # Compute target actions using the target actor network
            a2 = self._target_actor_net(obs2_norm)
            # Compute target Q-values using the target critic network
            q2 = self._target_critic_net(obs2_norm, a2)
            # Compute the target for the critic loss (Bellman equation)
            # Clip target Q-value to prevent divergence (common practice in Fetch envs)
            target_q = torch.clamp(rewards + constants.GAMMA * q2, -1. / (1. - constants.GAMMA), 0.)

        # Compute current Q-values using the main critic network
        current_q = self._critic_net(obs_norm, actions)
        # Calculate critic loss (Mean Squared Error)
        critic_loss = F.mse_loss(current_q, target_q)

        # Perform gradient descent step for the critic
        self.opt_c.zero_grad()
        critic_loss.backward()
        self.opt_c.step()

        # --- Actor Update ---
        # Compute actions for the current states using the main actor network
        actor_actions = self._actor_net(obs_norm)
        # Calculate actor loss: maximize Q-value (minimize negative Q-value)
        # Optional: add action magnitude regularization (L2 penalty)
        actor_loss = -self._critic_net(obs_norm, actor_actions).mean()
        # actor_loss += (actor_actions / self.ep_p['act_max']).pow(2).mean() # Example regularization

        # Perform gradient descent step for the actor
        self.opt_a.zero_grad()
        actor_loss.backward()
        self.opt_a.step()

    def soft_update(self):
        """ Performs Polyak (soft) updates for target networks. """
        # Update target actor
        for target_param, param in zip(self._target_actor_net.parameters(), self._actor_net.parameters()):
            target_param.data.mul_(constants.POLYAK).add_(param.data * (1. - constants.POLYAK))
        # Update target critic
        for target_param, param in zip(self._target_critic_net.parameters(), self._critic_net.parameters()):
            target_param.data.mul_(constants.POLYAK).add_(param.data * (1. - constants.POLYAK))

    def train(self):
        """ Main training loop for DDPG. """
        start_time = time.time()
        for epoch in range(constants.N_EPOCHS):
            epoch_start_time = time.time()
            # Loop over training cycles within an epoch
            for cycle in range(constants.N_CYCLES):
                # Collect experience
                self.collect_rollouts()
                # Perform gradient updates
                for _ in range(constants.N_BATCHES):
                    self.gradient_step()
                # Update target networks
                self.soft_update()

            # Evaluate the policy at the end of the epoch
            eval_success_rate, eval_reward = self.evaluate()
            epoch_duration = time.time() - epoch_start_time
            self.history_success_rate.append(eval_success_rate)
            self.history_reward.append(eval_reward)
            print(f"[DDPG] Epoch {epoch+1:3d}/{constants.N_EPOCHS} | Success Rate: {eval_success_rate:.3f} | Reward: {eval_reward:.3f} | Duration: {epoch_duration:.2f}s")

        total_duration = time.time() - start_time
        print(f"\nTraining finished. Total time: {total_duration:.2f}s")

    def save_model(self):
        """ Saves the DDPG model components. """
        print(f"Saving model to {self.model_path}.")
        checkpoint = {
            'actor_state_dict': self._actor_net.state_dict(),
            'critic_state_dict': self._critic_net.state_dict(),
            'o_norm_state': self.o_norm.get_state(),
            'g_norm_state': self.g_norm.get_state(),
            'history_success_rate': self.history_success_rate,
            'history_reward': self.history_reward
        }
        torch.save(checkpoint, self.model_path)
        print("Model saved.")

    def load_model(self):
        """ Loads the DDPG model components. """
        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found at {self.model_path}")
            return False
        try:
            print(f"Loading model from {self.model_path}.")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            self._actor_net.load_state_dict(checkpoint['actor_state_dict'])
            self._critic_net.load_state_dict(checkpoint['critic_state_dict'])
            self.o_norm.set_state(checkpoint['o_norm_state'])
            self.g_norm.set_state(checkpoint['g_norm_state'])
            self.history_success_rate = checkpoint.get('history_success_rate', [])
            self.history_reward = checkpoint.get('history_reward', [])

            # Ensure target networks are updated after loading
            self._target_actor_net = copy.deepcopy(self._actor_net)
            self._target_critic_net = copy.deepcopy(self._critic_net)

            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class SAC_HER(BaseAgent):
    """ Soft Actor-Critic (SAC) agent with HER. """
    def __init__(self, env, device):
        """ Initialises the SAC agent, networks, temperature, and optimizers. """
        super().__init__(env, "SAC", device) # Call base class constructor
        self.device = device
        p = self.ep_p # Environment parameters

        # Create actor (stochastic) and critic networks (two critics)
        self._actor_net = ActorStoch(p['obs'], p['g'], p['act'], p['act_max']).to(self.device)
        self._critic1_net = Critic(p['obs'], p['g'], p['act'], p['act_max']).to(self.device)
        self._critic2_net = Critic(p['obs'], p['g'], p['act'], p['act_max']).to(self.device)

        # Create target critic networks (SAC typically doesn't use a target actor)
        self._target_critic1_net = copy.deepcopy(self._critic1_net)
        self._target_critic2_net = copy.deepcopy(self._critic2_net)
        # Freeze target critic networks
        for param in self._target_critic1_net.parameters(): param.requires_grad = False
        for param in self._target_critic2_net.parameters(): param.requires_grad = False

        # Create optimizers for actor and critics
        self.opt_a = torch.optim.Adam(self._actor_net.parameters(), lr=constants.LR_ACTOR)
        self.opt_q1 = torch.optim.Adam(self._critic1_net.parameters(), lr=constants.LR_CRITIC)
        self.opt_q2 = torch.optim.Adam(self._critic2_net.parameters(), lr=constants.LR_CRITIC)

        # --- Temperature (alpha) setup for entropy regularization ---
        # Log alpha allows for unconstrained optimization, then exp() to get positive alpha
        self.log_alpha = torch.tensor(np.log(constants.ALPHA_INIT), dtype=torch.float32,
                                      device=self.device, requires_grad=True)
        # Optimizer for alpha
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=constants.LR_ALPHA)
        # Target entropy: usually set to -|Action Space Dimension|
        self.target_entropy = constants.TARGET_ENTROPY # Already set in BaseAgent.__init__

        # Add history storage
        self.history_success_rate = []
        self.history_reward = []

    @property
    def act(self):
        """ Returns the actor network instance. """
        return self._actor_net

    @property
    def alpha(self):
        """ Returns the current temperature value (alpha) by exponentiating log_alpha. """
        return self.log_alpha.exp()

    def gradient_step(self):
        """ Performs a single gradient update step for SAC. """
        # Sample a batch of transitions
        transitions = self.buf.sample(constants.BATCH_SZ)
        if transitions is None: return

        # Unpack and normalise
        o, o2, g, a, r = (transitions[k] for k in ['obs', 'obs_next', 'g', 'actions', 'r'])
        o_norm = self.o_norm.norm(o)
        o2_norm = self.o_norm.norm(o2)
        g_norm = self.g_norm.norm(g)

        # Convert to tensors
        obs_norm = torch.as_tensor(np.concatenate([o_norm, g_norm], axis=1), dtype=torch.float32, device=self.device)
        obs2_norm = torch.as_tensor(np.concatenate([o2_norm, g_norm], axis=1), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(r, dtype=torch.float32, device=self.device)

        # --- Critic Update ---
        with torch.no_grad():
            # Sample actions and get log probabilities for the next state from the current policy
            a2, logp2, _ = self._actor_net(obs2_norm)

            # Compute target Q-values using the minimum of the two target critics
            q1_target = self._target_critic1_net(obs2_norm, a2)
            q2_target = self._target_critic2_net(obs2_norm, a2)
            min_q_target = torch.min(q1_target, q2_target)

            # Add entropy term: Q_target = r + gamma * (min_Q_target - alpha * log_prob)
            # Clamp target Q-value like in DDPG/TD3
            target_q = torch.clamp(rewards + constants.GAMMA * (min_q_target - self.alpha * logp2),
                                   -1. / (1. - constants.GAMMA), 0.)

        # Compute current Q-values from both critics
        current_q1 = self._critic1_net(obs_norm, actions)
        current_q2 = self._critic2_net(obs_norm, actions)

        # Calculate critic loss (sum of MSE losses for both critics)
        loss_q = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Perform gradient descent step for both critics
        self.opt_q1.zero_grad()
        self.opt_q2.zero_grad()
        loss_q.backward()
        self.opt_q1.step()
        self.opt_q2.step()

        # Freeze critic gradients during actor and alpha updates
        for p in self._critic1_net.parameters(): p.requires_grad = False
        for p in self._critic2_net.parameters(): p.requires_grad = False

        # --- Actor Update ---
        # Sample new actions and log probabilities for the current state
        pi_new, logp_new, _ = self._actor_net(obs_norm)
        # Compute Q-values for the new actions using both critics
        q1_pi = self._critic1_net(obs_norm, pi_new)
        q2_pi = self._critic2_net(obs_norm, pi_new)
        min_q_pi = torch.min(q1_pi, q2_pi)

        # Calculate actor loss: J_pi = E[alpha * log_prob - min_Q]
        loss_a = (self.alpha.detach() * logp_new - min_q_pi).mean() # Detach alpha here

        # Perform gradient descent step for the actor
        self.opt_a.zero_grad()
        loss_a.backward()
        self.opt_a.step()

        # Unfreeze critic parameters
        for p in self._critic1_net.parameters(): p.requires_grad = True
        for p in self._critic2_net.parameters(): p.requires_grad = True

        # --- Temperature (Alpha) Update ---
        # Calculate alpha loss: J_alpha = E[-alpha * (log_prob + target_entropy)]
        # We optimize log_alpha, loss is -log_alpha * (log_prob + target_entropy)
        # Detach logp_new as we don't want to propagate gradients back to the policy here
        alpha_loss = -(self.log_alpha * (logp_new + self.target_entropy).detach()).mean()

        # Perform gradient descent step for log_alpha
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # --- Soft Target Updates for Critics ---
        with torch.no_grad():
            for tp, p in zip(self._target_critic1_net.parameters(), self._critic1_net.parameters()):
                tp.data.mul_(constants.POLYAK).add_(p.data * (1. - constants.POLYAK))
            for tp, p in zip(self._target_critic2_net.parameters(), self._critic2_net.parameters()):
                tp.data.mul_(constants.POLYAK).add_(p.data * (1. - constants.POLYAK))

    def train(self):
        """ Main training loop for SAC. """
        start_time = time.time()
        for epoch in range(constants.N_EPOCHS):
            epoch_start_time = time.time()
            # Loop over training cycles
            for cycle in range(constants.N_CYCLES):
                # Collect experience
                self.collect_rollouts()
                # Perform gradient updates
                for _ in range(constants.N_BATCHES):
                    self.gradient_step()

            # Evaluate the policy
            eval_success_rate, eval_reward = self.evaluate()
            epoch_duration = time.time() - epoch_start_time
            self.history_success_rate.append(eval_success_rate)
            self.history_reward.append(eval_reward)
            print(f"[SAC] Epoch {epoch+1:3d}/{constants.N_EPOCHS} | Success Rate: {eval_success_rate:.3f} | Reward: {eval_reward:.3f} | Duration: {epoch_duration:.2f}s")

        total_duration = time.time() - start_time
        print(f"\nTraining finished. Total time: {total_duration:.2f}s")

    def save_model(self):
        """ Saves the SAC model components. """
        print(f"Saving model to {self.model_path}.")
        checkpoint = {
            'actor_state_dict': self._actor_net.state_dict(),
            'critic1_state_dict': self._critic1_net.state_dict(),
            'critic2_state_dict': self._critic2_net.state_dict(),
            'log_alpha': self.log_alpha.item(),
            'o_norm_state': self.o_norm.get_state(),
            'g_norm_state': self.g_norm.get_state(),
            'history_success_rate': self.history_success_rate,
            'history_reward': self.history_reward,
        }
        torch.save(checkpoint, self.model_path)
        print("Model saved.")

    def load_model(self):
        """ Loads the SAC model components. """
        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found at {self.model_path}")
            return False
        try:
            print(f"Loading model from {self.model_path}.")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            self._actor_net.load_state_dict(checkpoint['actor_state_dict'])
            self._critic1_net.load_state_dict(checkpoint['critic1_state_dict'])
            self._critic2_net.load_state_dict(checkpoint['critic2_state_dict'])
            if 'log_alpha' in checkpoint:
                 self.log_alpha = torch.tensor(checkpoint['log_alpha'], dtype=torch.float32,
                                               device=self.device, requires_grad=True)
            self.o_norm.set_state(checkpoint['o_norm_state'])
            self.g_norm.set_state(checkpoint['g_norm_state'])
            self.history_success_rate = checkpoint.get('history_success_rate', [])
            self.history_reward = checkpoint.get('history_reward', [])

            # Update target networks
            self._target_critic1_net = copy.deepcopy(self._critic1_net)
            self._target_critic2_net = copy.deepcopy(self._critic2_net)

            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class TD3_HER(BaseAgent):
    """ Twin Delayed Deep Deterministic Policy Gradient (TD3) agent with HER. """
    def __init__(self, env, device):
        """ Initialises the TD3 agent, networks, and optimizers. """
        super().__init__(env, "TD3", device) # Call base class constructor
        self.device = device
        p = self.ep_p # Environment parameters

        # Create actor and critic networks (TD3 uses two critics)
        self._actor_net = ActorDet(p['obs'], p['g'], p['act'], p['act_max']).to(self.device)
        self._critic1_net = Critic(p['obs'], p['g'], p['act'], p['act_max']).to(self.device)
        self._critic2_net = Critic(p['obs'], p['g'], p['act'], p['act_max']).to(self.device)

        # Create target networks
        self._target_actor_net = copy.deepcopy(self._actor_net)
        self._target_critic1_net = copy.deepcopy(self._critic1_net)
        self._target_critic2_net = copy.deepcopy(self._critic2_net)
        # Freeze target networks
        for param in self._target_actor_net.parameters(): param.requires_grad = False
        for param in self._target_critic1_net.parameters(): param.requires_grad = False
        for param in self._target_critic2_net.parameters(): param.requires_grad = False

        # Create optimizers
        self.opt_a = torch.optim.Adam(self._actor_net.parameters(), lr=constants.LR_ACTOR)
        self.opt_q1 = torch.optim.Adam(self._critic1_net.parameters(), lr=constants.LR_CRITIC)
        self.opt_q2 = torch.optim.Adam(self._critic2_net.parameters(), lr=constants.LR_CRITIC)

        # Counter for policy delay updates
        self.total_it = 0

        # Add history storage
        self.history_success_rate = []
        self.history_reward = []

    @property
    def act(self):
        """ Returns the actor network instance. """
        return self._actor_net

    def gradient_step(self):
        """ Performs a single gradient update step for TD3. """
        self.total_it += 1 # Increment iteration counter

        # Sample a batch of transitions
        transitions = self.buf.sample(constants.BATCH_SZ)
        if transitions is None: return

        # Unpack and normalise
        o, o2, g, a, r = (transitions[k] for k in ['obs', 'obs_next', 'g', 'actions', 'r'])
        o_norm = self.o_norm.norm(o)
        o2_norm = self.o_norm.norm(o2)
        g_norm = self.g_norm.norm(g)

        # Convert to tensors
        obs_norm = torch.as_tensor(np.concatenate([o_norm, g_norm], axis=1), dtype=torch.float32, device=self.device)
        obs2_norm = torch.as_tensor(np.concatenate([o2_norm, g_norm], axis=1), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(r, dtype=torch.float32, device=self.device)

        # --- Critic Update ---
        with torch.no_grad():
            # Target policy smoothing: Add clipped noise to target actions
            noise = torch.clamp(
                constants.TARGET_NOISE * torch.randn_like(actions, device=self.device),
                -constants.NOISE_CLIP, constants.NOISE_CLIP
            )
            # Compute noisy target actions, clipped to valid range
            act2 = torch.clamp(
                self._target_actor_net(obs2_norm) + noise,
                -self.ep_p['act_max'], self.ep_p['act_max']
            )

            # Clipped double-Q learning: Compute target Q-values using the minimum of the two target critics
            q1_target = self._target_critic1_net(obs2_norm, act2)
            q2_target = self._target_critic2_net(obs2_norm, act2)
            min_q_target = torch.min(q1_target, q2_target)

            # Compute the final target for the critic loss
            target_q = torch.clamp(rewards + constants.GAMMA * min_q_target, -1. / (1. - constants.GAMMA), 0.)

        # Compute current Q-values from both critics
        current_q1 = self._critic1_net(obs_norm, actions)
        current_q2 = self._critic2_net(obs_norm, actions)

        # Calculate critic loss (sum of MSE losses for both critics)
        loss_q = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Perform gradient descent step for both critics
        self.opt_q1.zero_grad()
        self.opt_q2.zero_grad()
        loss_q.backward()
        self.opt_q1.step()
        self.opt_q2.step()

        # --- Delayed Policy Update ---
        if self.total_it % constants.POLICY_DELAY == 0:
            # Compute actions for the current states using the main actor
            actor_actions = self._actor_net(obs_norm)
            # Calculate actor loss: Use critic1 to evaluate the actor's actions
            actor_loss = -self._critic1_net(obs_norm, actor_actions).mean()

            # Perform gradient descent step for the actor
            self.opt_a.zero_grad()
            actor_loss.backward()
            self.opt_a.step()

            # --- Soft Target Updates (also delayed) ---
            # Update target actor
            for tp, p in zip(self._target_actor_net.parameters(), self._actor_net.parameters()):
                tp.data.mul_(constants.POLYAK).add_(p.data * (1. - constants.POLYAK))
            # Update target critics
            for tp, p in zip(self._target_critic1_net.parameters(), self._critic1_net.parameters()):
                tp.data.mul_(constants.POLYAK).add_(p.data * (1. - constants.POLYAK))
            for tp, p in zip(self._target_critic2_net.parameters(), self._critic2_net.parameters()):
                tp.data.mul_(constants.POLYAK).add_(p.data * (1. - constants.POLYAK))

    def train(self):
        """ Main training loop for TD3. """
        start_time = time.time()
        for epoch in range(constants.N_EPOCHS):
            epoch_start_time = time.time()
            # Loop over training cycles
            for cycle in range(constants.N_CYCLES):
                # Collect experience
                self.collect_rollouts()
                # Perform gradient updates
                for _ in range(constants.N_BATCHES):
                    self.gradient_step() # Includes delayed updates internally

            # Evaluate the policy
            eval_success_rate, eval_reward = self.evaluate()
            epoch_duration = time.time() - epoch_start_time
            self.history_success_rate.append(eval_success_rate)
            self.history_reward.append(eval_reward)
            print(f"[TD3] Epoch {epoch+1:3d}/{constants.N_EPOCHS} | Success Rate: {eval_success_rate:.3f} | Reward: {eval_reward:.3f} | Duration: {epoch_duration:.2f}s")

        total_duration = time.time() - start_time
        print(f"\nTraining finished. Total time: {total_duration:.2f}s")

    def save_model(self):
        """ Saves the TD3 model components. """
        print(f"Saving model to {self.model_path}.")
        checkpoint = {
            'actor_state_dict': self._actor_net.state_dict(),
            'critic1_state_dict': self._critic1_net.state_dict(),
            'critic2_state_dict': self._critic2_net.state_dict(),
            'o_norm_state': self.o_norm.get_state(),
            'g_norm_state': self.g_norm.get_state(),
            'history_success_rate': self.history_success_rate,
            'history_reward': self.history_reward,
        }
        torch.save(checkpoint, self.model_path)
        print("Model saved.")

    def load_model(self):
        """ Loads the TD3 model components. """
        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found at {self.model_path}")
            return False
        try:
            print(f"Loading model from {self.model_path}.")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            self._actor_net.load_state_dict(checkpoint['actor_state_dict'])
            self._critic1_net.load_state_dict(checkpoint['critic1_state_dict'])
            self._critic2_net.load_state_dict(checkpoint['critic2_state_dict'])
            self.o_norm.set_state(checkpoint['o_norm_state'])
            self.g_norm.set_state(checkpoint['g_norm_state'])
            self.history_success_rate = checkpoint.get('history_success_rate', [])
            self.history_reward = checkpoint.get('history_reward', [])

            # Update target networks
            self._target_actor_net = copy.deepcopy(self._actor_net)
            self._target_critic1_net = copy.deepcopy(self._critic1_net)
            self._target_critic2_net = copy.deepcopy(self._critic2_net)

            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
