"""This file contains a game between nonstationary sliding window classical TS bandit and model-based TS bandit."""

# Import packages
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import logging
import json


# Import bandit algorithms
from bandits import ThompsonSampling

# Import stationary models
from models import NonStationaryLogisticModel, NonStationaryClassicModel


# Initialize the environment
from envs import InsuranceMarketCt


logger = ut.get_logger(__name__)


# Read common parameters from yaml file
params = ut.read_parameters('common_parameters.yaml')
logger.info(json.dumps(params, indent=4))


# Define parameters
no_sim = 50  # Number of simulations
T = 10000  # Number of time steps
no_actions = 5  # Number of actions
action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)  # Action set

dimension = 2
tau = 1000

# Plotting parameters
foldername = 'images/ts'
filename = 'ts_mb_vs_ts_cl_ns_sw'
title = f'NS-SW Model-based TS vs Classic TS tau={tau}'

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
bandit_mb = ThompsonSampling(model=nonstationary_logistic_model)
# Initialize classic epsilon-greedy bandit
bandit_cl = ThompsonSampling(model=nonstationary_classic_model)


logger.info(f'Action set {action_set}')

# Run simulations
reward_history, bandit_mb_action_frequencies, bandit_cl_action_frequencies = ut.run_simulations(
    bandit_mb, bandit_cl, env, T, no_sim)


# Plot results using the plot functions from utils.py
ut.create_folder(foldername)
timesteps = np.arange(1, T+1)
start_plot_index = 10
title_mb = f'NS-SW Model-based TS'
title_cl = f'NS-SW Classic TS'
ut.plot_action_history_two_bandits(action_set, timesteps=timesteps, start_plot_index=start_plot_index,
                                   action_history1=bandit_mb_action_frequencies, action_history2=bandit_cl_action_frequencies,
                                   foldername=foldername, filename=filename,
                                   title1=title_mb, title2=title_cl)
ut.plot_reward_history(
    reward_history, bandit1_name='NS-SW Model-based TS',
    bandit2_name='NS-SW Classical TS', foldername=foldername, filename=filename, title=title)
