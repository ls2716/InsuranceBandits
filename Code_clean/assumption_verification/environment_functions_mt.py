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


def expected_reward_probability(env_params, S_i, S, x_samples, y_samples):
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
    prob = prob * norm.cdf(-(S_i - S_c + sigma*np.sqrt(1-rho) * x_samples
                             + sigma*np.sqrt(rho)*y_samples
                             ) / tau)
    for S_j in S:
        prob = prob * norm.cdf(-(S_i - S_j + sigma*rho *
                               x_samples) / np.sqrt((1-rho)*sigma**2))
    # Calculate the probability
    prob_ans = np.mean(prob, axis=1).flatten()
    true_margin = S_i + sigma*np.sqrt(1-rho) * x_samples \
        + sigma*np.sqrt(rho)*y_samples
    reward = np.mean(prob*true_margin, axis=1).flatten()
    return prob_ans, reward


# Define a function to compute S_i_star
def compute_S_i_star(S, env_params, x_samples, y_samples):
    """Compute the optimal S_i given S_j"""
    S_i = np.linspace(0, 2, 201)
    # print(S_i)
    prob, reward = expected_reward_probability(
        env_params, S_i, S, x_samples=x_samples, y_samples=y_samples)
    S_i_star = S_i[np.argmax(reward)]
    S_i = np.linspace(S_i_star-0.01, S_i_star+0.01, 201)
    prob, reward = expected_reward_probability(
        env_params, S_i, S, x_samples=x_samples, y_samples=y_samples)
    S_i_star = S_i[np.argmax(reward)]

    return S_i_star


# Compute derivative given S_j
def compute_derivative(S, index, env_params, x_samples, y_samples, dx=0.005):
    S0 = S.copy()
    S1 = S.copy()
    S0[index] = S0[index]-dx
    S1[index] = S1[index]+dx
    S_i_star0 = compute_S_i_star(S0, env_params, x_samples, y_samples)
    S_i_star1 = compute_S_i_star(S1, env_params, x_samples, y_samples)
    derivative = (S_i_star1-S_i_star0)/(2*dx)
    return derivative


if __name__ == "__main__":

    # Set seed for numpy
    np.random.seed(0)
    no_samples = 1000
    x_random_sample = np.random.normal(size=no_samples).reshape(1, -1)
    y_random_sample = np.random.normal(size=no_samples).reshape(1, -1)

    # Load the environment parameters
    # Read common parameters from yaml file
    params = ut.read_parameters('common_parameters.yaml')
    logger.info(json.dumps(params, indent=4))

    env_params = params["environment_parameters"]

    S = [0.5]

    S_i = np.linspace(-0.5, 1, 201)

    # Calculate the expected reward and probability
    prob, reward = expected_reward_probability(
        env_params, S_i, S, x_random_sample, y_random_sample)
    # Plot the results
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(S_i, prob, label='Probability')
    ax.plot(S_i, reward, label='Reward')
    ax.set_xlabel('S_i')
    ax.set_ylabel('Probability/Reward')
    ax.legend()
    ax.grid()
    plt.show()

    # tic = time.perf_counter()
    # for i in range(5):
    #     s_i_star = compute_S_i_star(
    #         S, env_params, x_samples=x_random_sample, y_samples=y_random_sample)
    # toc = time.perf_counter()
    # logger.info(f's_i_star = {s_i_star}')
    # logger.info(f'Elapsed time {toc-tic:.4f}s')

    dim = 5

    S_js = np.linspace(0, 0.2, dim)

    dSdis = np.zeros(shape=(dim, dim))
    dSdjs = np.zeros(shape=(dim, dim))
    sums = np.zeros(shape=(dim, dim))

    for i in range(dim):
        for j in range(dim):
            S = [S_js[i], S_js[j]]
            dSdi = compute_derivative(S, index=0, env_params=env_params,
                                      x_samples=x_random_sample, y_samples=y_random_sample)
            dSdj = compute_derivative(S, index=1, env_params=env_params,
                                      x_samples=x_random_sample, y_samples=y_random_sample)
            point = ', '.join([f'{item:.3f}' for item in S])
            derivatives = ', '.join([f'{item:.3f}' for item in [dSdi, dSdj]])
            logger.info(
                f'Derivatives at S=[{point}] are [{derivatives}]')
            dSdis[i, j] = dSdi
            dSdjs[i, j] = dSdj
            sums[i, j] = dSdi + dSdj

    print(' ')
    print(' ')
    print(' & ' + ' & '.join([f'{item:.2f}' for item in S_js]) + '\\\\')
    print('\\hline')
    for j in range(dim):
        print(f'{S_js[j]:.2f} & ' +
              ' & '.join([f'{sums[i,j]:.2f}' for i in range(dim)]) + '\\\\')

    print(' ')
    print(' ')
    print(' & ' + ' & '.join([f'{item:.2f}' for item in S_js]) + '\\\\')
    print('\\hline')
    for j in range(dim):
        print(f'{S_js[j]:.2f} & ' +
              ' & '.join([f'{dSdis[i,j]:.2f}' for i in range(dim)]) + '\\\\')
