from scipy.stats import norm
from scipy.integrate import trapz, trapezoid
import numpy as np
import matplotlib.pyplot as plt


def probability(S_i, S, S_c, sigma=0.2, rho=0, tau=0.2):
    interval = np.linspace(-50, 50, 1000)
    f = 1
    for S_j in S:
        f = f*norm.cdf(-(S_i - S_j + sigma * np.sqrt(1 - rho) * interval)
                       / (sigma * np.sqrt(1 - rho)))
    f = f * norm.cdf(-(S_i - S_c + sigma * np.sqrt(1 - rho) * interval)
                     / (np.sqrt(tau**2 + rho * sigma**2))) * norm.pdf(interval)
    return trapz(f, interval)


def reward(S_i, S, S_c, sigma=0.3, rho=0, tau=0.2):
    interval_x = np.linspace(-20, 20, 100).reshape(1, -1, 1)
    interval_y = np.linspace(-20, 20, 100).reshape(1, 1, -1)
    S_i = S_i.reshape(-1, 1, 1)
    f = 1
    for S_j in S:
        f = f*norm.cdf(-(S_i - S_j + sigma * np.sqrt(1 - rho) * interval_x)
                       / (sigma * np.sqrt(1 - rho)))
    f = f * norm.cdf(-(S_i - S_c + sigma * np.sqrt(1 - rho) * interval_x + sigma * np.sqrt(rho) * interval_y)
                     / (np.sqrt(tau**2 + rho * sigma**2))) \
        * (S_i + sigma * np.sqrt(1 - rho) * interval_x + sigma * np.sqrt(rho) * interval_y) \
        * norm.pdf(interval_x) * norm.pdf(interval_y)

    F_x = trapz(f, interval_x, axis=1)[None, :]
    F = trapz(F_x, interval_y, axis=2)
    return F.flatten()


def reward_previous(S_i, S, S_c, sigma=0.3, rho=0, tau=0.2):
    interval = np.linspace(-50, 50, 1000)
    S_i = S_i.reshape(-1, 1)
    f = 1
    for S_j in S:
        f = f*norm.cdf(-(S_i - S_j + sigma * np.sqrt(1 - rho) * interval)
                       / (sigma * np.sqrt(1 - rho)))
    f = norm.cdf(-(S_i - S_c + sigma * interval)
                 / (tau)) * (S_i) * norm.pdf(interval)
    reward = trapezoid(f, interval)
    return reward


def combined(S_i, S, S_c, sigma=0.3, rho=0, tau=0.2):
    interval_x = np.linspace(-20, 20, 100).reshape(1, -1, 1)
    interval_y = np.linspace(-20, 20, 100).reshape(1, 1, -1)
    S_i = S_i.reshape(-1, 1, 1)

    f = 1
    for S_j in S:
        f = f*norm.cdf(-(S_i - S_j + sigma * np.sqrt(1 - rho) * interval_x)
                       / (sigma * np.sqrt(1 - rho)))

    d_prob = f * norm.cdf(-(S_i - S_c + sigma * np.sqrt(1 - rho) * interval_x)
                          / np.sqrt(tau**2 + rho*sigma**2)) * norm.pdf(interval_x)

    prob = trapezoid(d_prob, interval_x, axis=1).flatten()

    reward_previous = trapezoid(d_prob * (S_i), interval_x, axis=1).flatten()

    dreward = f * norm.cdf(-(S_i - S_c + sigma * np.sqrt(1 - rho) * interval_x + sigma * np.sqrt(rho) * interval_y)
                           / (tau)) \
        * (S_i + sigma * np.sqrt(1 - rho) * interval_x + sigma * np.sqrt(rho) * interval_y) \
        * norm.pdf(interval_x) * norm.pdf(interval_y)
    reward_x = trapezoid(dreward, interval_x, axis=1)[None, :]
    reward = trapezoid(reward_x, interval_y, axis=2).flatten()

    return prob, reward, reward_previous


S_i = np.linspace(-1, 1, 10)
S_c = 0.5
S = [0.8]
# _ = reward_previous(S_i, S, S_c=S_c)
# prob, rew, rew_p = combined(S_i, S, S_c=S_c)

rew = reward(S_i, S, S_c)
