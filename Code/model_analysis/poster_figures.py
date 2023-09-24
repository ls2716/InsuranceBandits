import numpy as np
import matplotlib.pyplot as plt
import multi_mt as mm

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rc('axes', titlesize=16)


# Set up data
N = 200
x = np.linspace(-1., 2.5, N)

S_j = [1.2]
S_c = 1.
sigma = 0.3
tau = 0.2
rho = 0


y = mm.combined(x[:, None], S_j, S_c=S_c,
                sigma=sigma, tau=tau, rho=rho)

plt.figure(figsize=(6, 5))
plt.plot(x, y[0].flatten())
plt.plot([1.2, 1.2], [-0.2, 1.2], label='$S^{2}$')
plt.plot([S_c, S_c], [-0.2, 1.2], label='$S^C$')
plt.title('Offer acceptance probability')
plt.grid()
plt.legend()
plt.xlabel('$S^{(1)}$')
plt.ylabel('probability')
plt.ylim([-0.2, 1.2])
plt.xlim([-0.6, 2.1])
plt.tight_layout()
# plt.show()
plt.savefig('poster_probabilities.png', dpi=600)
plt.close()


plt.figure(figsize=(6, 5))
plt.plot(x, y[2].flatten())
plt.plot([1.2, 1.2], [-0.5, 1.], label='$S^{2}$')
plt.plot([S_c, S_c], [-0.5, 1.], label='$S^C$')
plt.title('Expected reward')
plt.grid()
plt.legend()
plt.xlabel('$S^{(1)}$')
plt.ylabel('expected reward')
plt.ylim([-0.5, 1.])
plt.xlim([-0.6, 2.1])
plt.tight_layout()
# plt.show()
plt.savefig('poster_rewards.png', dpi=600)
plt.close()
