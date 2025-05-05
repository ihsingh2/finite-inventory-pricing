# Finite Inventory Pricing

## Brief

Given a inventory of size $K$ and a booking window of $T$ days, learn an optimal pricing strategy for the underlying MDP with unknown transition probabilites.

## Instructions

### Steps to run

```shell
cd code
pip install -r requirements.txt
python main.py --help
python main.py --verbose --method=q-learning
python main.py --verbose --method=sarsa
# python main.py --verbose --method=value-iteration
# python main.py --verbose --method=policy-iteration
# python main.py --verbose --method=all
```

## File Structure

- ``code`` - Python code for modeling the MDP.
    - ``algorithms.py`` - Algorithms for solving the MDP.
    - ``constants.py`` - Parameters of the model, environment and algorithm.
    - ``main.py`` - Driver code invoking the algorithm.
    - ``model.py`` - Transitions and demand functions governing the model and environment.
    - ``test.ipynb`` - Code for experimental study.

- ``slides`` - Presentation explaining the problem formulation and experiment results.
