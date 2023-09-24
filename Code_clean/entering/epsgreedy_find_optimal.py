"""This script finds the parameters of the converged epsilon greedy bandit.
This is done by setting the fixed action agent to very high margin."""
"""This file tests the epsilon greedy bandit algorithm against
a fluctuating agent whose margin follows a sinusoid."""

# Import packages

# Import bandit algorithms

# Import fixed action agent


# Initialize the environment


import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os
from bandits import EpsGreedy
from bandits import FixedActionAgent
from models import NonStationaryClassicModel
from envs import InsuranceMarketCt
logger = ut.get_logger(__name__)

# Read common parameters from yaml file
params = ut.read_parameters('common_parameters.yaml')
logger.info(json.dumps(params, indent=4))


# Define parameters
no_sim = 40  # Number of simulations
T = 2000  # Number of time steps
no_actions = 5  # Number of actions
action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)  # Action set

epsilon = 0.05  # Epsilon for epsilon-greedy bandit
variance = 1.
gamma = 0.99

fixed_action = 1000.  # Action for fixed-action agent

# Plotting parameters
foldername = f'images/epsgreedy_entering_{no_actions}'
result_folder = f'results/epsgreedy_entering_{no_actions}'
filename = f'epsgreedy_only_{gamma}'
title = f'Ns-d {gamma} Classic Eps-Greedy only'


# Initialize the environment
env = InsuranceMarketCt(**params['environment_parameters'])  # Environment

# Initialize the model
model = NonStationaryClassicModel(variance=variance, candidate_margins=action_set,
                                  method='discounting', gamma=gamma)
# Initialize epsilon-greedy bandit
bandit = EpsGreedy(eps=epsilon, model=model)
# Initialize dummy agent
fixed_agent = FixedActionAgent(action=fixed_action)

logger.info(f'Action set {action_set}')


# Run simulations
reward_history, bandit_action_frequencies, fixed_action_frequencies = ut.run_simulations(
    bandit, fixed_agent, env, T, no_sim)

bandit.model_info()
bandit_filepath = f'entering/bandits/espgreedy_{gamma}.pkl'
ut.save_bandit(bandit, bandit_filepath)

# Print cumulative reward for the bandit and save to a file
logger.info(f'Sum of rewards for the bandit: {np.sum(reward_history[:, 0])}')
reward_1_sum = np.sum(reward_history[:, 0])
reward_2_sum = np.sum(reward_history[:, 1])
ut.create_folder(result_folder)
ut.print_result_to_file(reward_1_sum, reward_2_sum, os.path.join(
    result_folder, filename + '.txt'), f'Ns-d {gamma} Classic Eps-Greedy', 'Fixed-action Agent')

# Plot results using the plot functions from utils.py
ut.create_folder(foldername)
ut.plot_action_history(action_set, bandit_action_frequencies, foldername=foldername, filename=filename,
                       title=title)
ut.plot_smooth_reward_history(
    reward_history, bandit1_name=f'Ns-d {gamma} Classic Eps-Greedy',
    bandit2_name='Fixed-action agent ', foldername=foldername, filename=filename,
    title=title)
