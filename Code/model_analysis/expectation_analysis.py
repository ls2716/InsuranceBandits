from scipy.stats import norm
from scipy.integrate import trapz
import numpy as np
import matplotlib.pyplot as plt


def probability(S, S_c, sigma=0.2, rho=0, tau=0.2):
    interval = np.linspace(-50, 50, 1000)
    f = norm.cdf(-(S - S_c + sigma * np.sqrt(1 - rho) * interval)
                 / (np.sqrt(tau**2 + rho * sigma**2))) * norm.pdf(interval)
    return trapz(f, interval)


S_c = 0
S = np.linspace(S_c-2, S_c+2, 101)[:, None]


probs = probability(S, S_c)
fig, axs = plt.subplots(2, 1, figsize=(7, 9))
axs[0].plot(S, probs)
axs[0].set_title('Fill probability')
axs[0].set_xlabel('S')
axs[0].set_ylabel('Probability')
axs[1].plot(S.reshape(-1), S.reshape(-1)*probs.reshape(-1))
axs[1].set_title('Expectation')
axs[1].set_xlabel('S')
axs[1].set_ylabel('Expectation')
fig.suptitle(
    f'Reward probabilities for single quote \n with $S_c={S_c}$ $\\rho={0}$ $\\sigma=\\tau=0.2$')
# plt.show()
plt.savefig('reward_single.png')
plt.close()


# Derivative
def der_second(S, S_c, sigma=0.2, rho=0, tau=0.2):
    interval = np.linspace(-50, 50, 1000)
    f = norm.pdf(-(S - S_c + sigma * np.sqrt(1 - rho) * interval)
                 / (np.sqrt(tau**2 + rho * sigma**2))) * norm.pdf(interval)
    return S.reshape(-1)/np.sqrt(tau**2 + rho * sigma**2)*trapz(f, interval).reshape(-1)


probs = probability(S, S_c, sigma=0.2, tau=0.2)
second_term = der_second(S, S_c, sigma=0.2, tau=0.2)
fig, axs = plt.subplots(1, 1, figsize=(7, 5))
axs.plot(S, probs, label='first term')
axs.plot(S, second_term, label='- second term')
axs.plot(S, probs-second_term, label='sum')
axs.set_title('Expectation derivative')
axs.set_xlabel('S')
axs.set_ylabel('Expectation derivative')
axs.legend()
axs.grid()
# plt.show()
plt.savefig('derivative_single.png')
plt.close()
