"""This script contains a function which calculates the expected reward for the environment."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import logging

import json
import utils as ut

# Set up logger function
logger = ut.get_logger(__name__)
logger.setLevel(logging.INFO)


def expected_reward_probability(env_params, S_i, S, no_samples=1000):
    """Calculate the expected reward and probability
    given the environment parameters and agents' actions.
    """
    # Get parameters from the env_params dictionary
    sigma = env_params['sigma']
    tau = env_params['tau']
    rho = env_params['rho']
    S_c = env_params['S_c']
    S_i = S_i.reshape(-1, 1)
    # Generate Monte Carlo samples for normal distribution
    x_samples = np.random.normal(size=no_samples).reshape(1, -1)
    prob = np.ones_like(S_i)
    prob = prob * norm.cdf(-(S_i - S_c + sigma*np.sqrt(1-rho)
                           * x_samples) / np.sqrt(tau**2 + rho*sigma**2))
    for S_j in S:
        prob = prob * norm.cdf(-(S_i - S_j + sigma*rho *
                               x_samples) / np.sqrt((1-rho)*sigma**2))
    # Calculate the probability
    prob = np.mean(prob, axis=1).flatten()
    reward = prob*S_i.flatten()
    return prob, reward


if __name__ == "__main__":
    # Load the environment parameters
    # Read common parameters from yaml file
    params = ut.read_parameters('common_parameters.yaml')
    logger.info(json.dumps(params, indent=4))

    env_params = params["environment_parameters"]

    S_i = np.linspace(0, 2, 100)
    S = [0.5]
    # Calculate the expected reward and probability
    prob, reward = expected_reward_probability(env_params, S_i, S)
    # Plot the results
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(S_i, prob, label='Probability')
    ax.plot(S_i, reward, label='Reward')
    ax.set_xlabel('S_i')
    ax.set_ylabel('Probability/Reward')
    ax.legend()
    plt.show()
