"""This file contains the implementation of bandit algortihms
for the market problem.
"""
import numpy as np
import random
# Implement eps-greedy bandit


class ARCPricingModel(object):
    """ARC bandit algorithm as in Nash's paper
    """

    def __init__(self, candidate_margins, rho, beta, sigma) -> None:
        """Initialize the algorithm

        Arguments:
            - candidate_margins (numpy array of floats): list of possible margins
            - rho (float): scaling for lambda function
            - beta (float): discount factor
            - sigma (float): variance of measurements
        """
        self.actions = candidate_margins.flatten().reshape(-1, 1)
        self.K = self.actions.shape[0]
        self.setup_prior()
        self.rho = rho
        self.beta = beta
        self.sigma = sigma

    def setup_prior(self, variance=10**3, small_variance=1):
        """Setup uninformative prior with m=0
        and d diagonal array with big variance
        """
        self.m = np.zeros_like(self.actions)
        self.d = np.diag(np.ones_like(self.actions.flatten()))*small_variance
        self.d[0, 0] = variance

    def _f(self):
        """f function according to 3.2"""
        return self.m  # define f function

    def _norm(self, matrix):
        """returns max norm of a matrix"""
        return np.max(matrix.flatten())

    def _nu(self, a):
        """nu function"""
        self.lamda = self.rho*self._norm(self.d)
        a = a.reshape(-1, 1)
        expa = np.exp(a/self.lamda)
        return (expa/np.sum(expa, axis=0)).reshape(self.K, 1)

    def _eta(self, a):
        """eta function"""
        nu = self._nu(a)
        eta = -nu @ nu.T + np.diag(nu.flatten())
        return eta

    def reset(self):
        """Reset ARC algorithm"""
        self.setup_prior()

    def get_xk(self, k):
        """Get xk for given k
        - first k entries will be equal to 1"""
        x_k = np.zeros_like(self.m)
        for i in range(k+1):
            x_k[i, 0] = 1.
        return x_k

    def Var_mk(self, k):
        """Calculate variance of change in m if k-th arm
        is chosen.
        """
        x_k = self.get_xk(k)
        sigma_prime = (self.sigma + np.matmul(x_k.T,
                       np.matmul(self.d, x_k))).flatten()[0]

        D_tp1 = self.d - np.matmul(np.matmul(self.d, x_k) * 1/sigma_prime,
                                   np.matmul(x_k.T, self.d))
        sigma_mk = D_tp1 @ x_k/self.sigma*np.power(sigma_prime, 0.5)
        return sigma_mk @ sigma_mk.T

    def _Xi(self):
        """Calculate Xi matrix for the learning function"""
        eta = self._eta(self._f())
        return np.diag(np.diag(eta))

    def learning_function(self):
        """Evaluate the learning function
        """
        self.lamda = self.rho*self._norm(self.d)
        self.L = np.zeros_like(self.m)
        Xi = self._Xi()
        for k in range(self.K):
            self.L[k] = np.trace(Xi.T @ self.Var_mk(k))/self.lamda

    def get_probability(self):
        """Calculate probability using (a) variant

        That is, we use f to evaluate learning function, instead of
        finding the fixed point of the equation.
        """
        self.learning_function()
        # print(self._f())
        # print(self.L)
        self.U = self._nu(self._f() + self.beta/(1-self.beta)*self.L)
        return self.U

    def get_action(self):
        """Get actions according to batch size
        """
        action_index = np.random.choice(
            list(range(self.K)), p=self.get_probability().flatten())
        return self.actions.flatten()[action_index], action_index

    def update(self, k, reward):
        """Update posterior of the model parameters.
        """
        x_k = self.get_xk(k)
        sigma_prime = (self.sigma + np.matmul(x_k.T,
                       np.matmul(self.d, x_k))).flatten()[0]

        D_tp1 = self.d - np.matmul(np.matmul(self.d, x_k) * 1/sigma_prime,
                                   np.matmul(x_k.T, self.d))
        M_tp1 = self.m + D_tp1 @ x_k / self.sigma * \
            (reward - np.matmul(x_k.T, self.m))
        # self.d = D_tp1
        # self.m = M_tp1
        return M_tp1, D_tp1


if __name__ == "__main__":
    ...
