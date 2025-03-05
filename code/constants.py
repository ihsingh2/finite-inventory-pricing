# Model variables
K = 50                                      # Number of seats
T = 30                                      # Number of days
GAMMA = 0.99                                # Discount factor
PRICES = list(range(5000, 15001, 1000))     # List of prices

# Environment variables
DEMAND_MEAN_SCALE = 15.0                    # Scale in mean demand [scale * exp(-base * price)]
DEMAND_MEAN_BASE = 0.00025                  # Base in mean demand [scale * exp(-base * price)]
DEMAND_DISTRIBUTION = 'poisson'             # Distribution of demand for a given price
                                            # Options: ['poisson', 'binomial', 'geometric']

# Algorithm hyperparameters
NUM_ITERATIONS = 100                        # Number of iterations
NUM_EPISODES = 50000                        # Number of episodes to generate
EPSILON = 0.10                              # Parameter of epsilon-greedy strategy
TRACE_DECAY = 0.95                          # Decay eligibility trace by a scale after transition

# Miscellaneous
LOG_FREQUENCY = 10000                       # Log progress after a number of iterations
GAUSSIAN_SMOOTHING_STD = 100.0              # Smoothening parameter for per episode metrics
