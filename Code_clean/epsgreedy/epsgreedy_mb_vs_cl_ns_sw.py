"""This file contains a game between classical eps-greedy bandit and model-based eps-greedy bandit."""

# Import packages
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import logging
import json


# Import bandit algorithms
from bandits import EpsGreedy

# Import stationary models
from models import NonStationaryLogisticModel, NonStationaryClassicModel


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
epsilon = 0.05
tau = 300

# Plotting parameters
foldername = 'images/eps_greedy'
filename = 'eps_greedy_mb_ns_vs_epsgreedy_cl_ns_sw'
title = f'Nonstationary Model-based Epsilon Greedy vs Nonstationary Classic Epsilon Greedy\n eps={epsilon} tau={tau}'

# Initialize stationary model
nonstationary_logistic_model = NonStationaryLogisticModel(
    candidate_margins=action_set, dimension=dimension,
    method='sliding_window', tau=tau)
nonstationary_classic_model = NonStationaryClassicModel(
    variance=1, candidate_margins=action_set,
    method='sliding_window', tau=tau)
# Initialize the environment
env = InsuranceMarketCt(**params['environment_parameters'])  # Environment
# Initialize model-based epsilon-greedy bandit
bandit_mb = EpsGreedy(
    eps=epsilon, model=nonstationary_logistic_model)
# Initialize classic epsilon-greedy bandit
bandit_cl = EpsGreedy(
    eps=epsilon, model=nonstationary_classic_model)


logger.info(f'Action set {action_set}')

# Run simulations
reward_history, bandit_mb_action_frequencies, bandit_cl_action_frequencies = ut.run_simulations(
    bandit_mb, bandit_cl, env, T, no_sim)


# Plot results using the plot functions from utils.py
ut.create_folder(foldername)
timesteps = np.arange(1, T+1)
start_plot_index = 10
title_mb = f'Ns SW Model-based Epsilon Greedy epsilon={epsilon}'
title_cl = f'Ns SW Classic Epsilon Greedy epsilon={epsilon}'
ut.plot_action_history_two_bandits(action_set, timesteps=timesteps, start_plot_index=start_plot_index,
                                   action_history1=bandit_mb_action_frequencies, action_history2=bandit_cl_action_frequencies,
                                   foldername=foldername, filename=filename,
                                   title1=title_mb, title2=title_cl)
ut.plot_reward_history(
    reward_history, bandit1_name='Ns SW Model-based Epsilon Greedy',
    bandit2_name='Ns SW Classical Epsilon Greedy', foldername=foldername, filename=filename, title=title)
