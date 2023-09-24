import numpy as np
from scipy.integrate import trapz
from scipy.stats import norm
import matplotlib.pyplot as plt


S_i = np.linspace(-3, 3, 200)


def func(S_i, S, rho=0.5, sigma=0.1, tau=0.2, S_c=1):
    interval_x = np.linspace(-10, 10, 100).reshape(-1, 1, 1)
    interval_y = np.linspace(-10, 10, 100).reshape(1, -1, 1)
    S_i = S_i.reshape(1, 1, -1)
    f = 1
    for S_j in S:
        f = f*norm.cdf(-(S_i - S_j + sigma * np.sqrt(1 - rho) * interval_x)
                       / (sigma * np.sqrt(1 - rho)))
    f = f * norm.cdf(-(S_i - S_c + sigma * np.sqrt(1 - rho) * interval_x + sigma * np.sqrt(rho) * interval_y)
                     / (np.sqrt(tau**2 + rho * sigma**2))) \
        * (S_i + sigma * np.sqrt(1 - rho) * interval_x + sigma * np.sqrt(rho) * interval_y) \
        * norm.pdf(interval_x) * norm.pdf(interval_y)

    F_x = trapz(f, interval_x, axis=0)[None, :]
    F = trapz(F_x, interval_y, axis=1)
    return F.flatten()


f1 = func(S_i, [0.8], rho=0.2)
f2 = func(S_i, [0.8], rho=0.8)

plt.plot(S_i, f1, label='rho=0.2')
plt.plot(S_i, f2, label='rho=0.8')
plt.legend()
plt.grid()
plt.show()
