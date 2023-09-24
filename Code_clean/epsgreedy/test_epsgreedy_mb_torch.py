"""This file tests the model based epsilon greedy bandit algorithm against
a dummy agent which always quotes the same dummy action."""

# Import packages
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os

# Import bandit algorithms
from bandits import EpsGreedy

# Import stationary model
from models_torch import LogisticModelLaplace

# Import dummy agent
from bandits import FixedActionAgent

# Initialize the environment
from envs import InsuranceMarketCt


logger = ut.get_logger(__name__)


# Read common parameters from yaml file
params = ut.read_parameters('common_parameters.yaml')
logger.info(json.dumps(params, indent=4))


# Define parameters
no_sim = 50  # Number of simulations
T = 300  # Number of time steps
no_actions = 5  # Number of actions
action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)  # Action set

dimension = 2
epsilon = 0.05

fixed_action = 0.7  # Action for fixed-action agent

# Plotting parameters
foldername = f'images/eps_greedy_{no_actions}'
result_folder = f'results/eps_greedy_{no_actions}'
filename = 'eps_greedy_mb_vs_fixed_action'
title = f'Model-based Epsilon Greedy vs Fixed-Action Agent {fixed_action}'

# Initialize stationary model
stationary_model = LogisticModelLaplace(
    candidate_margins=action_set, dimension=dimension)
# Initialize the environment
env = InsuranceMarketCt(**params['environment_parameters'])  # Environment
# Initialize epsilon-greedy bandit
bandit = EpsGreedy(
    eps=epsilon, model=stationary_model)
# Initialize dummy agent
fixed_agent = FixedActionAgent(action=fixed_action)


logger.info(f'Action set {action_set}')

# Run simulations
reward_history, bandit_action_frequencies, fixed_action_frequencies = ut.run_simulations(
    bandit, fixed_agent, env, T, no_sim)

# Print cumulative reward for the bandit and save to a file
logger.info(f'Sum of rewards for the bandit: {np.sum(reward_history[:, 0])}')
reward_1_sum = np.sum(reward_history[:, 0])
reward_2_sum = np.sum(reward_history[:, 1])
ut.create_folder(result_folder)
ut.print_result_to_file(reward_1_sum, reward_2_sum, os.path.join(
    result_folder, filename + '.txt'), 'Model-based Eps-Greedy', 'Fixed-Action Agent')

# Plot results using the plot functions from utils.py
ut.create_folder(foldername)
ut.plot_action_history(action_set, bandit_action_frequencies, foldername=foldername, filename=filename,
                       title=title)
ut.plot_smooth_reward_history(
    reward_history, bandit1_name='Model-based Eps-Greedy',
    bandit2_name='Fixed-Action Agent', foldername=foldername, filename=filename, title=title)
