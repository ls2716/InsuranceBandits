from scipy.stats import norm
from scipy.integrate import trapz
import numpy as np
import matplotlib.pyplot as plt


m_c = 0
m_t = 0
tau = 0.1
P_i = np.linspace(m_c-2*tau, m_c+2*tau, 501)[:, None]


probs = norm.cdf(-(P_i-m_c)/tau)
fig, axs = plt.subplots(2, 1, figsize=(7, 9))
axs[0].plot(P_i, probs)
axs[0].set_title('Fill probability')
axs[0].set_xlabel('P_i')
axs[0].set_ylabel('Probability')
axs[1].plot(P_i, (P_i-m_t)*probs)
axs[1].set_title('Expectation')
axs[1].set_xlabel('P_i')
axs[1].set_ylabel('Expectation')
fig.suptitle(
    f'Reward probabilities for simple model')
# plt.show()
plt.savefig('simple_reward.png')
plt.close()


probs = norm.cdf(-(P_i-m_c)/tau)
second_term = - (P_i-m_t)/tau * norm.pdf(-(P_i-m_c)/tau)
fig, axs = plt.subplots(1, 1, figsize=(7, 5))
axs.plot(P_i, probs, label='first term')
axs.plot(P_i, -second_term, label='- second term')
axs.plot(P_i, probs+second_term, label='sum')
axs.set_title('Expectation derivative')
axs.set_xlabel('P_i')
axs.set_ylabel('Expectation derivative')
axs.legend()
axs.grid()
# plt.show()
plt.savefig('simple_derivative.png')
plt.close()


x = np.linspace(-3, 5, 1001)
f = norm.ppf(-(x)*norm.pdf(x))
fig, axs = plt.subplots(1, 1, figsize=(7, 5))
# axs.plot(x, (x+2)*norm.pdf(x)/norm.cdf(-x),
#          label='$y=(x+a) phi(x)$ for $a=-2$')
# axs.plot(x, (x)*norm.pdf(x),
#          label='$y=(-x+a) phi(-x)$ for $a=-2$')
# axs.plot(x, 0.5 * np.exp(-x**2/2), label='$y = 1/2*exp(-x^2/2))$')
axs.plot(x, norm.cdf(-x)*(-x) + norm.pdf(x), label='$y = \Phi(-x))$')
# axs.set_title('check for $x = \Phi^{-1}(x\phi(x))$ ')
axs.set_xlabel('x')
axs.legend()
axs.grid('minor')
plt.show()
plt.savefig('special_function.png')
plt.close()


x = np.linspace(-3, 5, 1001)
f = norm.ppf(-(x)*norm.pdf(x))
fig, axs = plt.subplots(1, 1, figsize=(7, 5))

axs.plot(x, (x-2)*norm.pdf(x) / (0.5 * np.exp(-(x**2)/2)),
         label='$y = (x+2)*norm.pdf(x)/ (1/2*exp(-x^2/2)))$')

axs.set_xlabel('x')
axs.legend()
axs.grid('minor')
# plt.show()
plt.savefig('special_function2.png')
plt.close()
