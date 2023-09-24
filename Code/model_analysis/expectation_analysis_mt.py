from scipy.stats import norm
from scipy.integrate import trapz
import numpy as np
import matplotlib.pyplot as plt


def probability(S, S_c, sigma=0.2, rho=0, tau=0.2):
    interval = np.linspace(-50, 50, 1000)
    f = norm.cdf(-(S - S_c + sigma * np.sqrt(1 - rho) * interval)
                 / (np.sqrt(tau**2 + rho * sigma**2))) * norm.pdf(interval)
    return trapz(f, interval)


def reward(S, S_c, sigma=0.3, rho=0, tau=0.2):
    interval = np.linspace(-50, 50, 1000)
    f = norm.cdf(-(S - S_c + sigma * np.sqrt(1 - rho) * interval)
                 / (np.sqrt(tau**2 + rho * sigma**2))) * (S+sigma*interval) * norm.pdf(interval)
    return trapz(f, interval)


def reward_previous(S, S_c, sigma=0.3, rho=0, tau=0.2):
    interval = np.linspace(-50, 50, 1000)
    f = norm.cdf(-(S - S_c + sigma * np.sqrt(1 - rho) * interval)
                 / (np.sqrt(tau**2 + rho * sigma**2))) * (S) * norm.pdf(interval)
    return trapz(f, interval)


S_c = 1
S = np.linspace(S_c-1, S_c+1, 101)[:, None]


probs = probability(S, S_c)
rewards = reward(S, S_c)
rewards_previous = reward_previous(S, S_c)
fig, axs = plt.subplots(2, 1, figsize=(7, 9))
axs[0].plot(S, probs)
axs[0].set_title('Fill probability')
axs[0].set_xlabel('S')
axs[0].set_ylabel('Probability')
axs[1].plot(S, rewards, label='new_reward')
# axs[1].plot(S, rewards_previous, label='previous_reward')
axs[1].set_title('Expectation')
axs[1].set_xlabel('S')
axs[1].set_ylabel('Expectation')
axs[1].legend()
axs[1].grid()
axs[0].grid()
fig.suptitle(
    f'Reward probabilities for single quote \n with $S_c={S_c}$ $\\rho={0}$ $\\sigma=0.3 \\tau=0.2$')
# plt.show()
plt.savefig('reward_single_mt.png')
plt.close()
