# Fetch Push and Reach

## Brief

Given a robotic manipulator with two-fingered gripper, move a block to a target position on a table by pushing it.

## Instructions

### Steps to run

```shell
cd code
pip install -r requirements.txt
python train.py
python eval.py
```

**Note:** Configure the hyperparameters in ``constants.py`` as needed.

## File Structure

- ``code`` - Python code for modeling the RL agent.
    - ``constants.py`` - Parameters of the model, environment and algorithm.
    - ``eval.py`` - Use the trained model to run a small simulation.
    - ``model.py`` - Algorithms and models for interaction with the environment.
    - ``plot_*.py`` - Code for experimental study.
    - ``train.py`` - Driver code for training the model.

- ``slides`` - Presentation explaining the problem formulation and experiment results.
