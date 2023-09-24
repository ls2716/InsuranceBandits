from scipy.stats import norm
from scipy.integrate import trapz
import numpy as np
import matplotlib.pyplot as plt


def probability(S, S_c, S_2=0, sigma=0.5, rho=0, tau=0.1):
    interval = np.linspace(-50, 50, 1000)
    f = norm.cdf(-(S - S_c + sigma * np.sqrt(1 - rho) * interval)
                 / (np.sqrt(tau**2 + rho * sigma**2)))*norm.cdf(-(S - S_2 + sigma * np.sqrt(1 - rho) * interval)
                                                                / (np.sqrt(sigma**2 - rho * sigma**2))) * norm.pdf(interval)
    return trapz(f, interval)


S_c = 0
S = np.linspace(S_c-2, S_c+3, 101)[:, None]


approximation = 1/(np.exp(3.5*(S-S_c))+np.exp(2.5*(S-0)) + 1)

probs = probability(S, S_c)
fig, axs = plt.subplots(1, 1, figsize=(9, 7))
axs.plot(S, probs, label='true')
axs.plot(S, approximation, label='approximation')
axs.plot(S.reshape(-1), probs.reshape(-1) -
         approximation.reshape(-1), label='difference')
axs.set_title('Fill probability')
axs.set_xlabel('S')
axs.set_ylabel('Probability')
plt.legend()
plt.savefig('approximation.png')
plt.close()
