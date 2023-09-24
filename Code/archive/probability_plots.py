import matplotlib.pyplot as plt

import numpy as np


X = np.linspace(-5, 10, 1000)


def func(x, a, b):
    probs = np.ones_like(x)
    zeros = np.where(x > a/b, 0, 1)
    func = -b*x + a
    probs = np.where(x > (a-1)/b, func, probs)
    probs = probs*zeros
    return probs


a = 1.5
b = 1

# plt.plot(X, func(X, a, b), label=f'probs a={a}, b={b}')

# plt.plot(X, func(X, a, b)*X, label=f'func a={a}, b={b}')
# plt.grid()
# plt.show()


def create_probability_sequence(X, centre, halfwidths):
    y = np.zeros_like(X)
    integral = 0
    for width in halfwidths:
        y_block = np.where(
            ((X > centre-width) & (X < centre + width)), 1., 0)
        y += y_block
        integral += width*2
    return y/integral


dx = X[1] - X[0]

S_j = 4.
widths = [0.5]
j_probs = create_probability_sequence(X, S_j, widths)


# S_is = np.copy(X).tolist()
# probs = []
# for S_i in S_is:
#     i_probs = create_probability_sequence(X, S_i, widths)
#     prob = 0
#     for j, j_prob in enumerate(j_probs):
#         prob += np.sum(np.where(X <= X[j], i_probs, 0.)*j_prob)*dx
#     probs.append(prob)

# plt.plot(S_is, probs)
# plt.grid()
# plt.show()


def f(X):
    y45 = 1 - (X-4)**2/2
    y46 = 0 + (6-X)**2/2
    probs = np.where(X > 6, 0, np.where(X < 4, 1., np.where(X < 5, y45, y46)))
    return probs


plt.plot(X, f(X))
plt.grid()
plt.show()

plt.plot(X, f(X)*X)
plt.plot(X, X * (1 - (X-4)**2/2))
plt.plot(X, X * ((6-X)**2/2))
plt.grid()
plt.show()
