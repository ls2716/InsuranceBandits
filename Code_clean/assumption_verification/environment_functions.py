import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import logging

import json
import utils as ut
import time

# Set up logger function
logger = ut.get_logger(__name__)
logger.setLevel(logging.INFO)


def expected_reward_probability(env_params, S_i, S, x_samples):
    """Calculate the expected reward and probability
    given the environment parameters and agents' actions.
    """
    # Get parameters from the env_params dictionary
    sigma = env_params['sigma']
    tau = env_params['tau']
    rho = env_params['rho']
    S_c = env_params['S_c']
    S_i = S_i.reshape(-1, 1)

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


# Define a function to compute S_i_star
def compute_S_i_star(S, env_params, random_sample):
    """Compute the optimal S_i given S_j"""
    S_i = np.linspace(0, 2, 201)
    # print(S_i)
    prob, reward = expected_reward_probability(
        env_params, S_i, S, x_samples=random_sample)
    S_i_star = S_i[np.argmax(reward)]
    S_i = np.linspace(S_i_star-0.01, S_i_star+0.01, 201)
    prob, reward = expected_reward_probability(
        env_params, S_i, S, x_samples=random_sample)
    S_i_star = S_i[np.argmax(reward)]

    return S_i_star


# Compute derivative given S_j
def compute_derivative(S, index, env_params, random_sample, dx=0.005):
    S0 = S.copy()
    S1 = S.copy()
    S0[index] = S0[index]-dx
    S1[index] = S1[index]+dx
    S_i_star0 = compute_S_i_star(S0, env_params, random_sample)
    S_i_star1 = compute_S_i_star(S1, env_params, random_sample)
    derivative = (S_i_star1-S_i_star0)/(2*dx)
    return derivative


if __name__ == "__main__":

    # Set seed for numpy
    np.random.seed(0)
    no_samples = 2000
    random_sample = np.random.normal(size=no_samples).reshape(1, -1)

    # Load the environment parameters
    # Read common parameters from yaml file
    params = ut.read_parameters('common_parameters.yaml')
    logger.info(json.dumps(params, indent=4))

    env_params = params["environment_parameters"]

    S = [0.5]

    tic = time.perf_counter()
    for i in range(5):
        s_i_star = compute_S_i_star(S, env_params, random_sample)
    toc = time.perf_counter()
    logger.info(f's_i_star = {s_i_star}')
    logger.info(f'Elapsed time {toc-tic:.4f}s')

    dim = 5

    S_js = np.linspace(-0.1, 0.1, dim)

    dSdis = np.zeros(shape=(dim, dim))
    dSdjs = np.zeros(shape=(dim, dim))
    sums = np.zeros(shape=(dim, dim))

    for i in range(dim):
        for j in range(dim):
            S = [S_js[i], S_js[j]]
            dSdi = compute_derivative(
                S, index=0, env_params=env_params, random_sample=random_sample)
            dSdj = compute_derivative(
                S, index=1, env_params=env_params, random_sample=random_sample)
            point = ', '.join([f'{item:.3f}' for item in S])
            derivatives = ', '.join([f'{item:.3f}' for item in [dSdi, dSdj]])
            logger.info(
                f'Derivatives at S=[{point}] are [{derivatives}]')
            dSdis[i, j] = dSdi
            dSdjs[i, j] = dSdj
            sums[i, j] = dSdi + dSdj

    # print(' & ' + ' & '.join([f'{item:.2f}' for item in S_js]) + '\\\\')
    # print('\\hline')
    # for i in range(dim):
    #     print(f'{S_js[i]:.2f} & ' +
    #           ' & '.join([f'{dSdis[i,j]:.2f}, {dSdjs[i,j]:.2f}, {sums[i,j]:.2f}' for j in range(dim)]) + '\\\\')

    # print(' ')
    # print('  ')
    # print(' & ' + ' & '.join([f'{item:.2f}' for item in S_js]) + '\\\\')
    # print('\\hline')
    # for i in range(dim):
    #     print(f'{S_js[i]:.2f} & ' +
    #           ' & '.join([f'{dSdis[i,j]:.2f}, {dSdjs[i,j]:.2f}' for j in range(dim)]) + '\\\\')
    print(' ')
    print('  ')
    print(' & ' + ' & '.join([f'{item:.2f}' for item in S_js]) + '\\\\')
    print('\\hline')
    for j in range(dim):
        print(f'{S_js[j]:.2f} & ' +
              ' & '.join([f'{sums[i,j]:.2f}' for i in range(dim)]) + '\\\\')

    print(' ')
    print('  ')
    print(' & ' + ' & '.join([f'{item:.2f}' for item in S_js]) + '\\\\')
    print('\\hline')
    for j in range(dim):
        print(f'{S_js[j]:.2f} & ' +
              ' & '.join([f'{dSdis[i,j]:.2f}' for i in range(dim)]) + '\\\\')

    print(' ')
    print('  ')
    print(' & ' + ' & '.join([f'{item:.2f}' for item in S_js]) + '\\\\')
    print('\\hline')
    for j in range(dim):
        print(f'{S_js[j]:.2f} & ' +
              ' & '.join([f'{dSdjs[i,j]:.2f}' for i in range(dim)]) + '\\\\')
