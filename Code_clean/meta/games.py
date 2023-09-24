"""This file contains games of non-stationary agents between each other"""

# Import packages
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os

from copy import deepcopy

# Initialize the environment
from envs import InsuranceMarketCt

from meta.agents import agent_dict, no_actions, action_set

logger = ut.get_logger(__name__)

# Read common parameters from yaml file
params = ut.read_parameters('common_parameters.yaml')
logger.info(json.dumps(params, indent=4))


# Define parameters
no_sim = 100  # Number of simulations
T = 1000  # Number of time steps
# no_actions = 5  # Number of actions
# action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)  # Action set


# Plotting parameters
base_foldername = f'images/meta_{no_actions}'
base_result_folder = f'results/meta_{no_actions}'


# Initialize the environment
env = InsuranceMarketCt(**params['environment_parameters'])  # Environment


# Define a simulation between two agents by names
def simulate_game(bandit_1_name, bandit_2_name):

    filename = f'{bandit_1_name}_vs_{bandit_2_name}'
    foldername = os.path.join(base_foldername, bandit_1_name)
    result_folder = os.path.join(base_result_folder, bandit_1_name)

    bandit_1 = deepcopy(agent_dict[bandit_1_name])
    bandit_2 = deepcopy(agent_dict[bandit_2_name])

    # Run simulations
    reward_history, bandit_1_action_frequencies, bandit_2_action_frequencies = ut.run_simulations(
        bandit_1, bandit_2, env, T, no_sim)

    # Print cumulative reward for the bandit and save to a file
    logger.info(
        f'Sum of rewards for the bandits: {np.sum(reward_history[:, 0])}, {np.sum(reward_history[:, 1])}')
    reward_1_sum = np.sum(reward_history[:, 0])
    reward_2_sum = np.sum(reward_history[:, 1])
    ut.create_folder(result_folder)
    ut.print_result_to_file(reward_1_sum, reward_2_sum, os.path.join(
        result_folder, filename + '.txt'), bandit_1_name, bandit_2_name)

    # Plot results using the plot functions from utils.py
    ut.create_folder(foldername)
    timesteps = np.arange(1, T+1)
    start_plot_index = 10
    ut.plot_action_history_two_bandits(action_set, timesteps=timesteps, start_plot_index=start_plot_index,
                                       action_history1=bandit_1_action_frequencies, action_history2=bandit_2_action_frequencies,
                                       foldername=foldername, filename=filename,
                                       title1=bandit_1_name, title2=bandit_2_name, show_plot=False)
    ut.plot_smooth_reward_history(
        reward_history, bandit1_name=bandit_1_name,
        bandit2_name=bandit_2_name, foldername=foldername, filename=filename, title=filename, show_plot=True)


agent_pairs = [
    ('epsgreedy_classic_ns_0.95', 'epsgreedy_classic_ns_0.95'),
    ('epsgreedy_classic_ns_0.95', 'epsgreedy_logistic_ns_0.95'),
    # ('epsgreedy_classic_ns_0.95', 'ucb_classic_ns_0.95'),
    # ('epsgreedy_classic_ns_0.95', 'ucb_logistic_ns_0.95'),
    # ('epsgreedy_classic_ns_0.95', 'ts_classic_ns_0.95'),
    # ('epsgreedy_classic_ns_0.95', 'ts_logistic_ns_0.95'),
    # ('epsgreedy_classic_ns_0.95', 'fluctuating_agent'),

    # ('epsgreedy_logistic_ns_0.95', 'epsgreedy_classic_ns_0.95'),
    ('epsgreedy_logistic_ns_0.95', 'epsgreedy_logistic_ns_0.95'),
    # ('epsgreedy_logistic_ns_0.95', 'ucb_classic_ns_0.95'),
    # ('epsgreedy_logistic_ns_0.95', 'ucb_logistic_ns_0.95'),
    # ('epsgreedy_logistic_ns_0.95', 'ts_classic_ns_0.95'),
    # ('epsgreedy_logistic_ns_0.95', 'ts_logistic_ns_0.95'),
    # ('epsgreedy_logistic_ns_0.95', 'fluctuating_agent'),

    # ('ucb_classic_ns_0.95', 'epsgreedy_classic_ns_0.95'),
    # ('ucb_classic_ns_0.95', 'epsgreedy_logistic_ns_0.95'),
    # ('ucb_classic_ns_0.95', 'ucb_classic_ns_0.95'),
    # ('ucb_classic_ns_0.95', 'ucb_logistic_ns_0.95'),
    # ('ucb_classic_ns_0.95', 'ts_classic_ns_0.95'),
    # ('ucb_classic_ns_0.95', 'ts_logistic_ns_0.95'),
    # ('ucb_classic_ns_0.95', 'fluctuating_agent'),

    # ('ucb_logistic_ns_0.95', 'epsgreedy_classic_ns_0.95'),
    # ('ucb_logistic_ns_0.95', 'epsgreedy_logistic_ns_0.95'),
    # ('ucb_logistic_ns_0.95', 'ucb_classic_ns_0.95'),
    # ('ucb_logistic_ns_0.95', 'ucb_logistic_ns_0.95'),
    # ('ucb_logistic_ns_0.95', 'ts_classic_ns_0.95'),
    # ('ucb_logistic_ns_0.95', 'ts_logistic_ns_0.95'),
    # ('ucb_logistic_ns_0.95', 'fluctuating_agent'),

    # ('ts_classic_ns_0.95', 'epsgreedy_classic_ns_0.95'),
    # ('ts_classic_ns_0.95', 'epsgreedy_logistic_ns_0.95'),
    # ('ts_classic_ns_0.95', 'ucb_classic_ns_0.95'),
    # ('ts_classic_ns_0.95', 'ucb_logistic_ns_0.95'),
    # ('ts_classic_ns_0.95', 'ts_classic_ns_0.95'),
    # ('ts_classic_ns_0.95', 'ts_logistic_ns_0.95'),
    # ('ts_classic_ns_0.95', 'fluctuating_agent'),

    # ('ts_logistic_ns_0.95', 'epsgreedy_classic_ns_0.95'),
    # ('ts_logistic_ns_0.95', 'epsgreedy_logistic_ns_0.95'),
    # ('ts_logistic_ns_0.95', 'ucb_classic_ns_0.95'),
    # ('ts_logistic_ns_0.95', 'ucb_logistic_ns_0.95'),
    # ('ts_logistic_ns_0.95', 'ts_classic_ns_0.95'),
    # ('ts_logistic_ns_0.95', 'ts_logistic_ns_0.95'),
    # ('ts_logistic_ns_0.95', 'fluctuating_agent'),
]


for agent_pair in agent_pairs:
    simulate_game(agent_pair[0], agent_pair[1])
