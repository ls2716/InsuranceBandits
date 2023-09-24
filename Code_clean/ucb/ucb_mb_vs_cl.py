"""This file contains a game between classical eps-greedy bandit and model-based eps-greedy bandit."""

# Import packages
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os

# Import bandit algorithms
from bandits import UCB_Bayes

# Import stationary models
from models import StationaryLogisticModel, StationaryClassicModel


# Initialize the environment
from envs import InsuranceMarketCt


logger = ut.get_logger(__name__)


# Read common parameters from yaml file
params = ut.read_parameters('common_parameters.yaml')
logger.info(json.dumps(params, indent=4))


# Define parameters
no_sim = 100  # Number of simulations
T = 1000  # Number of time steps
no_actions = 5  # Number of actions
action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)  # Action set

dimension = 2
quantile = 0.7

# Plotting parameters
foldername = f'images/ucb_{no_actions}'
result_folder = f'results/ucb_{no_actions}'
filename = 'ucb_mb_vs_ucb_cl'
title = f'Model-based UCB-Bayes vs Classic UCB-Bayes $q={quantile}$'

# Initialize stationary model
stationary_logistic_model = StationaryLogisticModel(
    candidate_margins=action_set, dimension=dimension)
stationary_classic_model = StationaryClassicModel(
    variance=1, candidate_margins=action_set)
# Initialize the environment
env = InsuranceMarketCt(**params['environment_parameters'])  # Environment
# Initialize model-based epsilon-greedy bandit
bandit_mb = UCB_Bayes(model=stationary_logistic_model,
                      T=None, quantile=quantile)
# Initialize classic epsilon-greedy bandit
bandit_cl = UCB_Bayes(model=stationary_classic_model,
                      T=None, quantile=quantile)


logger.info(f'Action set {action_set}')

# Run simulations
reward_history, bandit_mb_action_frequencies, bandit_cl_action_frequencies = ut.run_simulations(
    bandit_mb, bandit_cl, env, T, no_sim)


# Print cumulative reward for the bandit and save to a file
logger.info(
    f'Sum of rewards for the bandits: {np.sum(reward_history[:, 0])}, {np.sum(reward_history[:, 1])}')
reward_mb_sum = np.sum(reward_history[:, 0])
reward_cl_sum = np.sum(reward_history[:, 1])
ut.create_folder(result_folder)
ut.print_result_to_file(reward_mb_sum, reward_cl_sum, os.path.join(
    result_folder, filename + '.txt'), 'Model-based UCB-Bayes', 'Classic UCB-Bayes')

# Plot results using the plot functions from utils.py
ut.create_folder(foldername)
timesteps = np.arange(1, T+1)
start_plot_index = 10
title_mb = f'Model-based UCB-Bayes quantile={quantile}'
title_cl = f'Classic UCB-Bayes quantile={quantile}'
ut.plot_action_history_two_bandits(action_set, timesteps=timesteps, start_plot_index=start_plot_index,
                                   action_history1=bandit_mb_action_frequencies, action_history2=bandit_cl_action_frequencies,
                                   foldername=foldername, filename=filename,
                                   title1=title_mb, title2=title_cl)
ut.plot_smooth_reward_history(
    reward_history, bandit1_name='Model-based UCB-Bayes',
    bandit2_name='Classic UCB-Bayes', foldername=foldername, filename=filename, title=title)
