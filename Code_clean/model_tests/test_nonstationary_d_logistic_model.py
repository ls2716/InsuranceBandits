"""This file contains test of nonstationary posterior model with discounting.
"""
import numpy as np
import matplotlib.pyplot as plt

from models import NonStationaryLogisticModel, plot_logistic

import utils as ut
import json
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
gamma = 0.99

fixed_action = 0.75  # Action for fixed-action agent

# Plotting parameters
foldername = 'images/model_tests/nonstationary_logistic'
filename = 'discounting_test.png'
title = f'Non-stationary (discounting {gamma}) Logistic Model Test'

# Initialize stationary model
model = NonStationaryLogisticModel(
    candidate_margins=action_set, dimension=dimension,
    method='discounting', gamma=gamma)
# Initialize the environment
env = InsuranceMarketCt(**params['environment_parameters'])  # Environment


no_samples = 500
action_indices, rewards, observations, mean_rewards, frequencies = ut.get_reward_profile(
    env, no_samples, action_set, fixed_action=fixed_action)


logger.info(f'Rewards per action {mean_rewards}')
logger.info(f'Acceptance frequencies {frequencies}')


no_observations = 200
# Perform posterior updates
for it in range(no_observations):
    model.update(
        action_index=action_indices[it], reward=None, observation=observations[it])

    logger.debug(f'Posterior \n {model.mean.T} \n {model.cov}')


# Showing computed parameters
ut.create_folder(foldername)
plot_logistic(model.mean.reshape(-1), action_set,
              frequencies, title=f'{title} {fixed_action}', foldername=foldername, filename=filename)
