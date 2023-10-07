"""This file tests the ucb-bayes bandit algorithm against
a fluctuating agent whose margin follows a sinusoid."""

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
from bandits import FluctuatingActionAgent

from models import NonStationaryLogisticModel

# Initialize the environment
from envs import InsuranceMarketCt


logger = ut.get_logger(__name__)

# Read common parameters from yaml file
params = ut.read_parameters('common_parameters.yaml')
logger.info(json.dumps(params, indent=4))

nash_payoff = params['environment_parameters']['nash_payoff']
pareto_payoff = params['environment_parameters']['pareto_payoff']


def dimless_payoff(x):
    return (x - nash_payoff) / (pareto_payoff - nash_payoff)


# Define parameters
no_sim = 100  # Number of simulations
T = 1000  # Number of time steps
no_actions = 5  # Number of actions
action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)  # Action set

quantile = 0.7
variance = 1.
gamma = 0.95

mean_action = 0.6  # Action for fixed-action agent
half_amplitude = 0.3  # Half amplitude of the fluctuation
period = 500  # Period of the fluctuation

# Plotting parameters
foldername = f'images/ucb_{no_actions}'
result_folder = f'results/ucb_{no_actions}'
filename = f'ucb_bayes_mb_vs_fluc_action_{gamma}'
title = f'Ns-d {gamma} Model-based UCB-Bayes \nvs Fluctuating-Action Agent {mean_action},{half_amplitude},{period}'


# Initialize the environment
env = InsuranceMarketCt(**params['environment_parameters'])  # Environment

# Initialize the model
model = NonStationaryLogisticModel(
    candidate_margins=action_set, method='discounting', gamma=gamma)
# Initialize epsilon-greedy bandit
bandit = UCB_Bayes(model=model, T=None, c=0, quantile=quantile)
# Initialize dummy agent
fixed_agent = FluctuatingActionAgent(mean_action=mean_action,
                                     half_amplitude=half_amplitude,
                                     period=period)


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
    result_folder, filename + '.txt'), f'Ns-d {gamma} Model-based UCB-Bayes', 'Fluctuating-Action Agent')


# Plot results using the plot functions from utils.py
ut.create_folder(foldername)
ut.plot_action_history(action_set, bandit_action_frequencies, foldername=foldername, filename=filename,
                       title=title)
ut.plot_smooth_reward_history(
    dimless_payoff(reward_history), bandit1_name=f'Ns-d {gamma} Model-based UCB-Bayes',
    bandit2_name='Fluctuating-Action Agent', foldername=foldername, filename=filename,
    title=title)
