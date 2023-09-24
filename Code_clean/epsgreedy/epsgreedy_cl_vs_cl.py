"""This file contains a game between classical eps-greedy bandit
 and another classical eps-greedy bandit."""

# Import packages
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import logging
import json


# Import bandit algorithms
from bandits import EpsGreedy

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
T = 2000  # Number of time steps
no_actions = 9  # Number of actions
action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)  # Action set

dimension = 2
epsilon = 0.05


# Plotting parameters
foldername = 'images/eps_greedy_9'
filename = 'eps_greedy_cl_vs_epsgreedy_cl'
title = f'Classic Epsilon Greedy vs Classic Epsilon Greedy eps={epsilon}'

# Initialize stationary model
stationary_classic_model_1 = StationaryClassicModel(
    variance=1, candidate_margins=action_set)
stationary_classic_model_2 = StationaryClassicModel(
    variance=1, candidate_margins=action_set)
# Initialize the environment
env = InsuranceMarketCt(**params['environment_parameters'])  # Environment
# Initialize model-based epsilon-greedy bandit
bandit_cl_1 = EpsGreedy(
    eps=epsilon, model=stationary_classic_model_1)
# Initialize classic epsilon-greedy bandit
bandit_cl_2 = EpsGreedy(
    eps=epsilon, model=stationary_classic_model_2)


logger.info(f'Action set {action_set}')

# Run simulations
reward_history, bandit_cl_1_action_frequencies, bandit_cl_2_action_frequencies = ut.run_simulations(
    bandit_cl_1, bandit_cl_2, env, T, no_sim)


# Plot results using the plot functions from utils.py
ut.create_folder(foldername)
timesteps = np.arange(1, T+1)
start_plot_index = 10
title_cl_1 = f'Classic 1 Epsilon Greedy epsilon={epsilon}'
title_cl_2 = f'Classic 2 Epsilon Greedy epsilon={epsilon}'
ut.plot_action_history_two_bandits(action_set, timesteps=timesteps, start_plot_index=start_plot_index,
                                   action_history1=bandit_cl_1_action_frequencies, action_history2=bandit_cl_2_action_frequencies,
                                   foldername=foldername, filename=filename,
                                   title1=title_cl_1, title2=title_cl_2)
ut.plot_reward_history(
    reward_history, bandit1_name='Classic 1 Epsilon Greedy',
    bandit2_name='Classic 2 Epsilon Greedy', foldername=foldername, filename=filename, title=title)
