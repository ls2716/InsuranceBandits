"""This script checks the assumptions on the reward function."""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.integrate import trapezoid

from functools import partial

# Define the reward function


def compute_prob_reward(S_i, S, S_c, sigma=0.3, rho=0, tau=0.2):
    """Reward function using Monte Carlo integration."""
    # Define 1000 x samples
    x_samples = np.random.normal(size=(20000))
    y_samples = np.random.normal(size=(20000))

    S_i = S_i.reshape(-1, 1)

    f = 1
    for S_j in S:
        f = f*norm.cdf(-(S_i - S_j + sigma * np.sqrt(1 - rho) * x_samples)
                       / (sigma * np.sqrt(1 - rho)))

    d_prob = f * norm.cdf(-(S_i - S_c + sigma * np.sqrt(1 - rho) * x_samples)
                          / np.sqrt(tau**2 + rho*sigma**2))

    prob = np.mean(d_prob, axis=1)

    reward_previous = np.mean(d_prob * (S_i), axis=1).flatten()

    dreward = f * norm.cdf(-(S_i - S_c + sigma * np.sqrt(1 - rho) * x_samples + sigma * np.sqrt(rho)
                             * y_samples) / (tau)) \
        * (S_i + sigma * np.sqrt(1 - rho) * x_samples + sigma * np.sqrt(rho) * y_samples)
    reward = np.mean(dreward, axis=1)

    return prob, reward, reward_previous


def compute_reward(S_i, S, S_c, sigma=0.3, rho=0, tau=0.2):
    prob, reward, reward_previous = compute_prob_reward(
        S_i, S, S_c, sigma, rho, tau)
    return reward_previous


# If run as main script
if __name__ == '__main__':

    # Define the parameters
    S_c = 2.
    S = np.array([])
    sigma = 0.1
    rho = 0
    tau = 0.2

    # Test the reward function
    S_i = np.linspace(0., 1.0, 1001)

    # prob, rewards, rewards_previous = compute_prob_reward(
    #     S_i, S, S_c, sigma, rho, tau)

    # # Plot the results
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].plot(S_i, prob, label='Probability')
    # axs[1].plot(S_i, rewards, label='Reward')
    # axs[1].plot(S_i, rewards_previous, label='Reward Previous')

    # axs[0].set_xlabel('S_i')
    # axs[1].set_xlabel('S_i')
    # axs[0].set_ylabel('Probability')
    # axs[1].set_ylabel('Reward')
    # # Show legend
    # axs[0].legend()
    # axs[1].legend()
    # # Show grid
    # axs[0].grid()
    # axs[1].grid()

    # plt.show()

    # Define partial function for best response
    def best_response_gradients(S, h=0.05):
        """Compute the best response gradients."""

        gradients = np.zeros_like(S)
        for j, s_j in enumerate(S):
            S_prime = np.copy(S)
            S_prime[j] = S_prime[j] + h
            s_i_best_prime = S_i[np.argmax(
                compute_reward(S_i, S_prime, S_c, sigma, rho, tau))]
            S_bis = np.copy(S)
            S_bis[j] = S_bis[j] - h
            s_i_best_bis = S_i[np.argmax(
                compute_reward(S_i, S_bis, S_c, sigma, rho, tau))]
            gradients[j] = (s_i_best_prime - s_i_best_bis)/(2*h)

        return gradients

    # # Check gradients function
    # S = np.array([0.5, 1.0])
    # gradients = best_response_gradients(S)
    # print(gradients)

    # no_points = 7
    # S_1 = np.linspace(-0.1, 0.5, no_points, endpoint=True)
    # S_2 = np.linspace(-0.1, 0.5, no_points, endpoint=True)

    # gradient_sums = np.zeros((S_1.shape[0], S_2.shape[0]))

    # for i in range(S_1.shape[0]):
    #     for j in range(i, S_2.shape[0]):
    #         S = np.array([S_1[i], S_2[j]])
    #         gradients = best_response_gradients(S, h=0.025)
    #         print('S:', S)
    #         print('gradients:', gradients,
    #               f'sum: {np.sum(gradients):.2f}')
    #         gradient_sums[i, j] = np.sum(gradients)

    # print(gradient_sums)

    # alpha = 0.2
    deltas = np.linspace(0., 0.5, 11)
    S_0 = np.array([-0.6])
    direction = np.array([1.])
    S_i_bests = np.zeros_like(deltas)
    for i in range(deltas.shape[0]):
        print('Iteration', i+1, 'of', deltas.shape[0])
        S = S_0 + deltas[i]*direction
        print(S)
        S_i_bests[i] = S_i[np.argmax(
            compute_reward(S_i, S, S_c, sigma, rho, tau))]
        print(S_i_bests[i])
    print(S_i_bests)
    # Plot the results
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.plot(deltas, S_i_bests, label='S_i_best')

    axs.set_xlabel('$\Delta$')
    axs.set_ylabel('$S_i^B$')
    axs.set_aspect('equal')
    axs.set_xlim([-0.1, 1.1])
    axs.set_ylim([-0.1, 1.1])
    axs.grid('both')
    plt.show()
