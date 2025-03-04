import math
import numpy as np
from constants import \
        K, T, GAMMA, PRICES, NUM_ITERATIONS, NUM_EPISODES, EPSILON, TRACE_DECAY, LOG_FREQUENCY
from model import sample_demand, expected_reward, transition_prob

def value_iteration(verbose=False):
    """ Value Iteration algorithm for the finite horizon problem. """

    # Initialize value function and policy
    V = {}
    P = {}
    for inv in range(K + 1):
        for t in range(T + 1):
            V[(inv, t)] = 0.0
            P[(inv, t)] = PRICES[0]

    # Iterate over timesteps, after the zero reward terminal state (i.e., 0 days left)
    for t in range(1, T + 1):

        # Log progress
        if verbose:
            print(f"Evaluating for t={t}")

        # Iterate over available seats (states)
        for inv in range(1, K + 1):

            # Find the optimal price (action)
            V_max = float("-inf")
            P_opt = PRICES[0]
            for price in PRICES:
                exp_value = 0
                exp_reward = expected_reward(inv, t, price)
                for next_inv in range(inv + 1):
                    prob = transition_prob(inv, t, price, next_inv, t - 1)
                    exp_value += prob * V[(next_inv, t - 1)]
                V_new = exp_reward + GAMMA * exp_value
                if V_max < V_new:
                    V_max = V_new
                    P_opt = price
            V[(inv, t)] = V_max
            P[(inv, t)] = P_opt

    return V, P

def policy_iteration(verbose=False):
    """ Policy Iteration algorithm for the finite horizon problem. """

    # Initialize value function and policy
    V = {}
    P = {}
    for inv in range(K + 1):
        for t in range(T + 1):
            V[(inv, t)] = 0.0
            P[(inv, t)] = PRICES[0]

    # Iterate over different policies
    for iteration in range(NUM_ITERATIONS):

        # Log progress
        if verbose:
            print(f"Evaluating for iteration {iteration}")

        # =============== Policy Evaluation ===============
        # Iterate over timesteps, after the zero reward terminal state (i.e., 0 days left)
        for t in range(1, T + 1):

            # Iterate over available seats (states)
            for inv in range(1, K + 1):

                # Evaluate the value for the current policy
                V_old = V[(inv, t)]
                price = P[(inv, t)]
                exp_value = 0
                exp_reward = expected_reward(inv, t, price)
                for next_inv in range(inv + 1):
                    prob = transition_prob(inv, t, price, next_inv, t - 1)
                    exp_value += prob * V[(next_inv, t - 1)]
                V[(inv, t)] = exp_reward + GAMMA * exp_value

        # =============== Policy Improvement ===============
        converged = True

        # Iterate over timesteps, after the zero reward terminal state (i.e., 0 days left)
        for t in range(1, T + 1):

            # Iterate over available seats (states)
            for inv in range(1, K + 1):

                # Find the optimal price (action)
                V_max = float('-inf')
                P_old = P[(inv, t)]
                P_opt = PRICES[0]
                for price in PRICES:
                    exp_value = 0
                    exp_reward = expected_reward(inv, t, price)
                    for next_inv in range(inv + 1):
                        prob = transition_prob(inv, t, price, next_inv, t - 1)
                        exp_value += prob * V[(next_inv, t - 1)]
                    V_new = exp_reward + GAMMA * exp_value
                    if V_max < V_new:
                        V_max = V_new
                        P_opt = price
                P[(inv, t)] = P_opt

                # Convergence criterion
                if P_old != P_opt:
                    converged = False

        # Stopping condition
        if converged:
            break

    return V, P

def q_learning(verbose=False):
    """ Q-Learning algorithm for the finite horizon problem. """

    # Initialize state action value function, policy and eligibility trace
    Q = {}
    P = {}
    for inv in range(K + 1):
        for t in range(T + 1):
            for price in PRICES:
                Q[(inv, t, price)] = 0.0
            P[(inv, t)] = PRICES[0]

    # Initialize list to store metrics
    episode_rewards = []
    episode_regrets = []
    cumulative_rewards = [0.0, ]
    cumulative_regrets = [0.0, ]

    # Optimal value for regret calculation
    V_opt, _ = value_iteration()

    # Iterate over episodes
    for episode in range(NUM_EPISODES):

        # Log progress
        if verbose and episode % LOG_FREQUENCY == 0 and episode > 0:
            print(f"Evaluating for episode {episode}")

        # Learning rate for current episode (https://arxiv.org/abs/2110.15093)
        alpha = math.ceil(10 / (episode + 1))

        # Initialize state and reward
        starting_inventory = np.random.randint(1, K+1)
        inventory = starting_inventory
        time = T
        episode_reward = 0

        # Iterate over step of episode
        while time > 0 and inventory > 0:

            # Choose an epsilon-greedy action
            if np.random.random() < EPSILON:
                price = np.random.choice(PRICES)
            else:
                price = P[(inventory, time)]

            # Take action and observe next state and reward
            demand = sample_demand(price)
            sales = min(demand, inventory)
            reward = price * sales
            episode_reward = reward + GAMMA * episode_reward
            next_inventory = inventory - sales
            next_time = time - 1

            # Update state value function
            Q_next = max(Q[(next_inventory, next_time, p)] for p in PRICES)
            Q[(inventory, time, price)] = (1 - alpha) * Q[(inventory, time, price)] + \
                                                        alpha * (reward + GAMMA * Q_next)

            # Update policy
            Q_max = float('-inf')
            P_max = PRICES[0]
            for p in PRICES:
                q = Q[(inventory, time, p)]
                if Q_max < q:
                    Q_max = q
                    P_max = p
            P[(inventory, time)] = P_max

            # Update current state
            inventory = next_inventory
            time = next_time

            # Terminal state
            if inventory == 0 or time == 0:
                break

        # Calculate regret
        episode_rewards.append(episode_reward)
        episode_regrets.append(V_opt[(starting_inventory, T)] - episode_reward)
        cumulative_rewards.append(cumulative_rewards[-1] + episode_rewards[-1])
        cumulative_regrets.append(cumulative_regrets[-1] + episode_regrets[-1])

    return Q, P, {
        'episode_rewards': episode_rewards,
        'episode_regrets': episode_regrets,
        'cumulative_rewards': cumulative_rewards[1:],
        'cumulative_regrets': cumulative_regrets[1:],
    }

def sarsa_lambda(verbose=False):
    """ SARSA(λ) algorithm for the finite horizon problem. """

    # Initialize state action value function and policy
    Q = {}
    P = {}
    for inv in range(K + 1):
        for t in range(T + 1):
            for price in PRICES:
                Q[(inv, t, price)] = 0.0
            P[(inv, t)] = PRICES[0]

    # Initialize list to store metrics
    episode_rewards = []
    episode_regrets = []
    cumulative_rewards = [0.0, ]
    cumulative_regrets = [0.0, ]

    # Optimal value for regret calculation
    V_opt, _ = value_iteration()

    for episode in range(NUM_EPISODES):

        # Log progress
        if verbose and episode % LOG_FREQUENCY == 0 and episode > 0:
            print(f"Evaluating for episode {episode}")

        # Learning rate for current episode (https://arxiv.org/abs/2110.15093)
        alpha = math.ceil(10 / (episode + 1))

        # Initialize state, reward and eligibility trace
        starting_inventory = np.random.randint(1, K+1)
        inventory = starting_inventory
        time = T
        episode_reward = 0
        E = {}

        # Choose an epsilon-greedy action
        if np.random.random() < EPSILON:
            price = np.random.choice(PRICES)
        else:
            price = P[(inventory, time)]

        # Iterate over step of episode
        while time > 0 and inventory > 0:

            # Take action and observe next state and reward
            demand = sample_demand(price)
            sales = min(demand, inventory)
            reward = price * sales
            episode_reward = reward + GAMMA * episode_reward
            next_inventory = inventory - sales
            next_time = time - 1

            # Choose an epsilon-greedy next action
            if np.random.random() < EPSILON:
                next_price = np.random.choice(PRICES)
            else:
                next_price = P[(next_inventory, next_time)]

            # Calculate TD error
            td_error = reward + GAMMA * Q[(next_inventory, next_time, next_price)] - Q[(inventory, time, price)]

            # Update state action value function and eligibility trace
            E[(inventory, time, price)] = \
                    E[(inventory, time, price)] + 1 if (inventory, time, price) in E.keys() else 1
            for state_action in list(E.keys()):
                Q[state_action] += alpha * td_error * E[state_action]
                E[state_action] *= GAMMA * TRACE_DECAY

                # Remove small traces to save memory
                if E[state_action] < 0.01:
                    del E[state_action]

            # Update policy
            Q_max = float('-inf')
            P_max = PRICES[0]
            for p in PRICES:
                q = Q[(inventory, time, p)]
                if Q_max < q:
                    Q_max = q
                    P_max = p
            P[(inventory, time)] = P_max

            # Update current state and action
            inventory = next_inventory
            time = next_time
            price = next_price

            # Terminal state
            if inventory == 0 or time == 0:
                break

        # Calculate regret
        episode_rewards.append(episode_reward)
        episode_regrets.append(V_opt[(starting_inventory, T)] - episode_reward)
        cumulative_rewards.append(cumulative_rewards[-1] + episode_rewards[-1])
        cumulative_regrets.append(cumulative_regrets[-1] + episode_regrets[-1])

    return Q, P, {
        'episode_rewards': episode_rewards,
        'episode_regrets': episode_regrets,
        'cumulative_rewards': cumulative_rewards[1:],
        'cumulative_regrets': cumulative_regrets[1:],
    }
