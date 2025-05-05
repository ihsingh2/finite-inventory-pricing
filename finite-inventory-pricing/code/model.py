import math
import numpy as np
from constants import K, T, DEMAND_MEAN_SCALE, DEMAND_MEAN_BASE, DEMAND_DISTRIBUTION

def demand_function(price):
    """ Returns the mean for the demand distribution for a given price. """

    return DEMAND_MEAN_SCALE * math.exp(-DEMAND_MEAN_BASE * price)

def probability_mass_fn(mean, x):
    """ Calculates the probability of sampling x from a Poisson distribution of given mean. """

    if DEMAND_DISTRIBUTION == 'poisson':
        return (math.exp(-mean) * mean**x) / math.factorial(x)

    if DEMAND_DISTRIBUTION == 'binomial':
        return math.comb(K, x) * (mean / K)**x * (1 - (mean / K))**(K - x)

    if DEMAND_DISTRIBUTION == 'geometric':
        return (1 - (1 / mean))**(x - 1) * (1 / mean)

def sample_demand(price):
    """ Samples a demand at the given price from the distribution. """

    mean = demand_function(price)

    if DEMAND_DISTRIBUTION == 'poisson':
        return np.random.poisson(mean)

    if DEMAND_DISTRIBUTION == 'binomial':
        return np.random.binomial(K, mean / K)

    if DEMAND_DISTRIBUTION == 'geometric':
        return np.random.geometric(1 / mean)

def expected_reward(inventory, time, price):
    """ Calculates the expected reward for a given state (inventory, time) and action (price). """

    # Variables
    cum_prob = 0
    expected_reward = 0
    lam = demand_function(price)

    # Iterate over all possible small demands
    for d in range(inventory):
        prob = probability_mass_fn(lam, d)
        expected_reward += prob * price * d
        cum_prob += prob

    # Reward for clearing the entire inventory
    expected_reward += (1 - cum_prob) * price * inventory

    return expected_reward

def transition_prob(inventory, time, price, next_inventory, next_time):
    """ Calculate the probability of transitioning from (inventory, time) to (next_inventory, next_time) given a price. """

    # Only transitions resulting in decrease of inventory and a unit drop in time are allowed
    if next_time != time - 1 or next_inventory > inventory:
        return 0.0

    # Inventory cannot drop below zero
    if inventory == next_inventory == 0:
        return 1.0

    # Average demand for price
    lam = demand_function(price)

    # Probability of demand clearing the inventory
    if next_inventory == 0 and inventory > 0:
        return 1 - sum(probability_mass_fn(lam, d) for d in range(inventory))

    # Probability of smaller demands
    demand = inventory - next_inventory
    return probability_mass_fn(lam, demand)
