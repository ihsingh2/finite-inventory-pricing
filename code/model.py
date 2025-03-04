import math
import numpy as np
from constants import K, T, DEMAND_MEAN_SCALE, DEMAND_MEAN_BASE

def poisson_pmf(mean, x):
    """ Calculates the probability of sampling x from a Poisson distribution of given mean. """

    return (math.exp(-mean) * mean**x) / math.factorial(x)

def demand_function(price):
    """ Returns the mean (lambda parameter) for the Poisson distribution based on price. """

    return DEMAND_MEAN_SCALE * math.exp(-DEMAND_MEAN_BASE * price)

def sample_demand(price):
    """ Samples a demand at the given price from Poisson distribution. """

    lam = demand_function(price)
    return np.random.poisson(lam)

def expected_reward(inventory, time, price):
    """ Calculates the expected reward for a given state (inventory, time) and action (price). """

    # Variables
    cum_prob = 0
    expected_reward = 0
    lam = demand_function(price)

    # Iterate over all possible small demands
    for d in range(inventory):
        prob = poisson_pmf(lam, d)
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
        return 1 - sum(poisson_pmf(lam, d) for d in range(inventory))

    # Probability of smaller demands
    demand = inventory - next_inventory
    return poisson_pmf(lam, demand)
