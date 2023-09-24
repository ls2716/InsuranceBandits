"""This file tests the model-based ucb-bayes bandit algorithm against
a dummy agent which always quotes the same fixed action."""

# Import packages
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os

# Import bandit algorithms
from bandits import EpsGreedy, UCB_Bayes

# Import fixed action agent
from bandits import FixedActionAgent

from models import StationaryLogisticModel

# Initialize the environment
from envs import InsuranceMarketCt


logger = ut.get_logger(__name__)

# Read common parameters from yaml file
params = ut.read_parameters('common_parameters.yaml')
logger.info(json.dumps(params, indent=4))


# Define parameters
no_sim = 100  # Number of simulations
T = 1000  # Number of time steps
no_actions = 129  # Number of actions
action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)  # Action set

dimension = 2
quantile = 0.7

fixed_action = 0.7  # Action for fixed-action agent

# Plotting parameters
foldername = f'images/ucb_{no_actions}'
result_folder = f'results/ucb_{no_actions}'
filename = 'ucb_bayes_mb_vs_fixed_action'
title = f'Model-based UCB-Bayes vs Fixed-Action Agent {fixed_action}'


# Initialize the environment
env = InsuranceMarketCt(**params['environment_parameters'])  # Environment

# Initialize the model
model = StationaryLogisticModel(candidate_margins=action_set)
# Initialize epsilon-greedy bandit
bandit = UCB_Bayes(model=model, T=None, c=0, quantile=quantile)
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
    result_folder, filename + '.txt'), 'Model-based UCB-Bayes', 'Fixed-Action Agent')

# Plot results using the plot functions from utils.py
ut.create_folder(foldername)
ut.plot_action_history(action_set, bandit_action_frequencies, foldername=foldername, filename=filename,
                       title=title)
ut.plot_smooth_reward_history(
    reward_history, bandit1_name='Model-based UCB-Bayes',
    bandit2_name='Fixed-Action Agent', foldername=foldername, filename=filename,
    title=title)
