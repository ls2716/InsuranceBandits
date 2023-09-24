"""This script simulates the entrance of a model-based agent to an environment
with a classic epsilon-greedy bandit agent who already learned the optimal parameters 
of the environment.
"""

import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os
from bandits import EpsGreedy
from bandits import FixedActionAgent
from models import NonStationaryClassicModel, NonStationaryLogisticModel
from envs import InsuranceMarketCt
logger = ut.get_logger(__name__)

# Read common parameters from yaml file
params = ut.read_parameters('common_parameters.yaml')
logger.info(json.dumps(params, indent=4))


# Define parameters
no_sim = 50  # Number of simulations
T = 1000  # Number of time steps
no_actions = 5  # Number of actions
action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)  # Action set

epsilon = 0.05  # Epsilon for epsilon-greedy bandit
variance = 1.
gamma = 0.99


# Plotting parameters
foldername = f'images/epsgreedy_entering_{no_actions}'
result_folder = f'results/epsgreedy_entering_{no_actions}'
filename = f'epsgreedy_mb_enters_cl_{gamma}'
title = f'Ns-d {gamma} Model-base Eps-Greedy enteting Classic Eps-Greedy'


# Initialize the environment
env = InsuranceMarketCt(**params['environment_parameters'])  # Environment

# Initialize the model
classic_model = NonStationaryClassicModel(variance=variance, candidate_margins=action_set,
                                          method='discounting', gamma=gamma)
# Initialize the model
logistic_model = NonStationaryLogisticModel(candidate_margins=action_set,
                                            method='discounting', gamma=gamma)
# Initialize epsilon-greedy bandit
bandit_cl = EpsGreedy(eps=epsilon, model=classic_model)
# Initialize dummy agent
bandit_mb = EpsGreedy(eps=epsilon, model=logistic_model)

bandit_filepath = f'entering/bandits/espgreedy_{gamma}.pkl'
bandit1_0 = ut.load_bandit(bandit_filepath)
bandit1_0.model_info()


logger.info(f'Action set {action_set}')

# Run simulations
reward_history, bandit_cl_action_frequencies, bandit_mb_action_frequencies = ut.run_simulations(
    bandit_cl, bandit_mb, env, T, no_sim, bandit1_0=bandit1_0)


# Print cumulative reward for the bandit and save to a file
logger.info(f'Sum of rewards for the bandit: {np.sum(reward_history[:, 0])}')
reward_1_sum = np.sum(reward_history[:, 0])
reward_2_sum = np.sum(reward_history[:, 1])
ut.create_folder(result_folder)
ut.print_result_to_file(reward_1_sum, reward_2_sum, os.path.join(
    result_folder, filename + '.txt'), f'Ns-d {gamma} Classic Eps-Greedy', f'Ns-d {gamma} Model-based Eps-Greedy')

# Plot results using the plot functions from utils.py
ut.create_folder(foldername)
timesteps = np.arange(1, T+1)
start_plot_index = 10
ut.plot_action_history_two_bandits(action_set, timesteps=timesteps, start_plot_index=start_plot_index,
                                   action_history1=bandit_cl_action_frequencies, action_history2=bandit_mb_action_frequencies,
                                   foldername=foldername, filename=filename,
                                   title1=f'Ns-d {gamma} Classic Eps-Greedy', title2=f'Ns-d {gamma} Model-based Eps-Greedy', show_plot=True)
ut.plot_smooth_reward_history(
    reward_history, bandit1_name=f'Ns-d {gamma} Classic Eps-Greedy',
    bandit2_name='Model-base Eps-greedy', foldername=foldername, filename=filename,
    title=title)
