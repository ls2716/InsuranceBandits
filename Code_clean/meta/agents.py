""" Definition of agents for meta games """
import numpy as np
import logging
import json
import os

# Import bandit algorithms
from bandits import EpsGreedy, UCB_Bayes, ThompsonSampling
# Import fluctuating action agent
from bandits import FluctuatingActionAgent

# Import models
from models import NonStationaryLogisticModel, NonStationaryClassicModel

# Import environment
from envs import InsuranceMarketCt


logger = logging.getLogger(__name__)


# Initialize a fluctuating action agent
mean_action = 0.6  # Action for fixed-action agent
half_amplitude = 0.3  # Half amplitude of the fluctuation
period = 500  # Period of the fluctuation
fluctuating_agent = FluctuatingActionAgent(
    mean_action=mean_action,
    half_amplitude=half_amplitude,
    period=period,
)

# Define hyperparameters for agents
epsilon = 0.05
quantile = 0.7

variance = 1.

no_actions = 129
action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)


def get_agent(bandit: str, model: str, gamma: float):
    if model == 'classic':
        model = NonStationaryClassicModel(
            variance,
            action_set,
            method='discounting',
            gamma=gamma
        )
    elif model == 'logistic':
        model = NonStationaryLogisticModel(
            action_set,
            method='discounting',
            gamma=gamma
        )
    else:
        raise ValueError('Incorrect model. Choose "classic" or "logistic".')

    if bandit == 'epsgreedy':
        agent = EpsGreedy(eps=epsilon, model=model)
    elif bandit == 'ucb':
        agent = UCB_Bayes(model=model, T=None, c=0, quantile=quantile)
    elif bandit == 'ts':
        agent = ThompsonSampling(model)

    return agent


def get_agent_sw(bandit: str, model: str, tau: int):
    if model == 'classic':
        model = NonStationaryClassicModel(
            variance,
            action_set,
            method='sliding_window',
            tau=tau
        )
    elif model == 'logistic':
        model = NonStationaryLogisticModel(
            action_set,
            method='sliding_window',
            tau=tau
        )
    else:
        raise ValueError('Incorrect model. Choose "classic" or "logistic".')

    if bandit == 'epsgreedy':
        agent = EpsGreedy(eps=epsilon, model=model)
    elif bandit == 'ucb':
        agent = UCB_Bayes(model=model, T=None, c=0, quantile=quantile)
    elif bandit == 'ts':
        agent = ThompsonSampling(model)

    return agent


# Define a dictionary with agents
agent_dict = {
    'epsgreedy_classic_ns_0.70': get_agent('epsgreedy', 'classic', 0.70),
    'epsgreedy_classic_ns_0.80': get_agent('epsgreedy', 'classic', 0.80),
    'epsgreedy_classic_ns_0.85': get_agent('epsgreedy', 'classic', 0.85),
    'epsgreedy_classic_ns_0.90': get_agent('epsgreedy', 'classic', 0.9),
    'epsgreedy_classic_ns_0.93': get_agent('epsgreedy', 'classic', 0.93),
    'epsgreedy_classic_ns_0.95': get_agent('epsgreedy', 'classic', 0.95),
    'epsgreedy_classic_ns_0.97': get_agent('epsgreedy', 'classic', 0.97),
    'epsgreedy_classic_ns_0.99': get_agent('epsgreedy', 'classic', 0.99),
    'epsgreedy_classic_ns_0.995': get_agent('epsgreedy', 'classic', 0.995),
    'epsgreedy_classic_ns_1.0': get_agent('epsgreedy', 'classic', 1.),
    'epsgreedy_logistic_ns_0.80': get_agent('epsgreedy', 'logistic', 0.8),
    'epsgreedy_logistic_ns_0.90': get_agent('epsgreedy', 'logistic', 0.9),
    'epsgreedy_logistic_ns_0.93': get_agent('epsgreedy', 'logistic', 0.93),
    'epsgreedy_logistic_ns_0.95': get_agent('epsgreedy', 'logistic', 0.95),
    'epsgreedy_logistic_ns_0.99': get_agent('epsgreedy', 'logistic', 0.99),
    'epsgreedy_logistic_ns_1.0': get_agent('epsgreedy', 'logistic', 1.),
    'ucb_classic_ns_0.90': get_agent('ucb', 'classic', 0.9),
    'ucb_classic_ns_0.93': get_agent('ucb', 'classic', 0.93),
    'ucb_classic_ns_0.95': get_agent('ucb', 'classic', 0.95),
    'ucb_classic_ns_0.97': get_agent('ucb', 'classic', 0.97),
    'ucb_classic_ns_0.99': get_agent('ucb', 'classic', 0.99),
    'ucb_classic_ns_0.995': get_agent('ucb', 'classic', 0.995),
    'ucb_classic_ns_1.0': get_agent('ucb', 'classic', 1.),
    'ucb_logistic_ns_0.95': get_agent('ucb', 'logistic', 0.95),
    'ucb_logistic_ns_0.99': get_agent('ucb', 'logistic', 0.99),
    'ts_classic_ns_0.95': get_agent('ts', 'classic', 0.95),
    'ts_classic_ns_0.99': get_agent('ts', 'classic', 0.99),
    'ts_logistic_ns_0.95': get_agent('ts', 'logistic', 0.95),
    'ts_logistic_ns_0.99': get_agent('ts', 'logistic', 0.99),
    'fluctuating_agent': fluctuating_agent,

    'epsgreedy_classic_nssw_200': get_agent_sw('epsgreedy', 'classic', 200),
    'epsgreedy_logistic_nssw_200': get_agent_sw('epsgreedy', 'logistic', 200),
    'epsgreedy_classic_nssw_100': get_agent_sw('epsgreedy', 'classic', 100),
    'epsgreedy_logistic_nssw_100': get_agent_sw('epsgreedy', 'logistic', 100),
    'epsgreedy_classic_nssw_75': get_agent_sw('epsgreedy', 'classic', 75),
    'epsgreedy_logistic_nssw_75': get_agent_sw('epsgreedy', 'logistic', 75),
    'epsgreedy_classic_nssw_50': get_agent_sw('epsgreedy', 'classic', 50),
    'epsgreedy_logistic_nssw_50': get_agent_sw('epsgreedy', 'logistic', 50),
    'epsgreedy_classic_nssw_40': get_agent_sw('epsgreedy', 'classic', 40),
    'epsgreedy_logistic_nssw_40': get_agent_sw('epsgreedy', 'logistic', 40),
    'epsgreedy_classic_nssw_30': get_agent_sw('epsgreedy', 'classic', 30),
    'epsgreedy_logistic_nssw_30': get_agent_sw('epsgreedy', 'logistic', 30),
    'epsgreedy_classic_nssw_20': get_agent_sw('epsgreedy', 'classic', 20),
    'epsgreedy_logistic_nssw_20': get_agent_sw('epsgreedy', 'logistic', 20),
    'epsgreedy_classic_nssw_10': get_agent_sw('epsgreedy', 'classic', 10),
    'epsgreedy_logistic_nssw_10': get_agent_sw('epsgreedy', 'logistic', 10),

    'ucb_classic_nssw_200': get_agent_sw('ucb', 'classic', 200),
    'ucb_logistic_nssw_200': get_agent_sw('ucb', 'logistic', 200),
    'ucb_classic_nssw_100': get_agent_sw('ucb', 'classic', 100),
    'ucb_logistic_nssw_100': get_agent_sw('ucb', 'logistic', 100),
    'ucb_classic_nssw_50': get_agent_sw('ucb', 'classic', 50),
    'ucb_logistic_nssw_50': get_agent_sw('ucb', 'logistic', 50),
    'ucb_classic_nssw_30': get_agent_sw('ucb', 'classic', 30),
    'ucb_logistic_nssw_30': get_agent_sw('ucb', 'logistic', 30),
    'ucb_classic_nssw_20': get_agent_sw('ucb', 'classic', 20),
    'ucb_logistic_nssw_20': get_agent_sw('ucb', 'logistic', 20),
    'ucb_classic_nssw_10': get_agent_sw('ucb', 'classic', 10),
    'ucb_logistic_nssw_10': get_agent_sw('ucb', 'logistic', 10),
}
