"""This file contains the implementation of bandit algortihms
for the market problem.
"""
import numpy as np
import random
from scipy.stats import norm
from scipy.special import softmax, expit
import scipy
# Implement eps-greedy bandit


class TSPricingModel(object):
    """ARC bandit algorithm as in Nash's paper
    """

    def __init__(self, candidate_margins, rho, beta, mc_size=1000, dimension=2) -> None:
        """Initialize the algorithm

        Arguments:
            - candidate_margins (numpy array of floats): list of possible margins
            - rho (float): scaling for lambda function
            - beta (float): discount factor
            - sigma (float): variance of measurements
        """
        # if dimension != 2:
        #     raise ValueError('Dimension has to be equal to 2.')
        self.actions = candidate_margins.flatten().reshape(-1, 1)
        self.K = self.actions.shape[0]
        self.dimension = dimension
        self.setup_prior()
        self.rho = rho
        self.beta = beta
        self.mc_size = mc_size
        self.f_lambda = np.pi/5.35  # lambda for sigmoid approximation
        self.new_step()

        self.action_xks = []
        for action in range(self.K):
            xk = self.sk_vector(action)
            self.action_xks.append(xk)
        self.action_xks = np.hstack(self.action_xks)

    def setup_prior(self, variance=10**2):
        """Setup uninformative prior with m=0
        and d diagonal array with big variance
        """
        self.m = np.zeros(shape=(self.dimension, 1))
        self.d = np.diag(np.ones_like(self.m.flatten())*variance)
        self.d0 = self.d  # Saving initial variance
        self.m0 = np.zeros(shape=(self.dimension, 1))

        self.xks = None
        self.wks = None
        self.Ys = None
        self.t = 0

    def new_step(self):
        """Clean all used variables to indicate the new step and recalculate all the variables
        """
        self.f = None
        self.lamda = None

    def sk_vector(self, k):
        """Return a vector with entries [1, s, s^2, ...] for given sk"""
        sk = self.actions[k, 0]
        sk_vector = np.ones_like(self.m)
        for i in range(1, self.dimension):
            sk_vector[i, 0] = sk_vector[i-1, 0]*sk
        return sk_vector

    def _norm(self, matrix):
        """returns max norm of a matrix"""
        return np.max(matrix.flatten())

    def _lamda(self):
        """Compute lamda for entropy regularization"""
        if self.lamda is not None:
            return self.lamda
        self.lamda = self.rho*self._norm(self.d)
        return self.lamda

    def _nu(self, a):
        """nu function"""
        a = a.reshape(-1, 1)
        a = a/self._lamda()
        nu = softmax(a).reshape(self.K, 1)
        return nu

    def reset(self):
        """Reset ARC algorithm"""
        self.setup_prior()

    def sigmoid(self, x):
        """Sigmoid function"""
        return expit(x)

    def dtp1(self, k):
        """Compute new sigma"""
        dtp1 = self.d + self.E_dk(k)
        return dtp1

    def get_rewards(self):
        """Get action using Thomson sampling"""
        # Sample parameters from the posterior
        # m_sample = np.random.multivariate_normal(self.m.flatten(), self.d)
        # Compute expected rewards using sampled parameters
        rewards = self.sigmoid(self.m.T @ self.action_xks)
        rewards = self.action_xks[1, :] * rewards
        return rewards

    def get_probability(self):
        """Calculate probability using (a) variant

        That is, we use f to evaluate learning function, instead of
        finding the fixed point of the equation.
        """
        self.U = self._nu(self.get_rewards())  # Without learning function
        return self.U

    def get_action(self):
        """Get actions according to batch size
        """
        action_index = np.random.choice(
            list(range(self.K)), p=self.get_probability().flatten())
        return self.actions.flatten()[action_index], action_index

    def update_all_observations(self, k, Y):
        """Calculate current posterior based on all the observations"""
        # self.m = self.m0
        xk = self.sk_vector(k)
        if self.xks is None:
            self.xks = xk
        else:
            self.xks = np.hstack((self.xks, xk))
        if self.Ys is None:
            self.Ys = np.array([Y])
        else:
            self.Ys = np.append(self.Ys, Y)
        self.t += 1
        if self._norm(self.d) > 50:
            self.m = self.m0 * 0
        mutxks = self.m.reshape(-1) @ self.xks
        self.sigma_sum = np.zeros_like(self.d0)
        self.sigma_sum_factors = self.sigmoid(mutxks)*(
            1-self.sigmoid(mutxks))
        self.sigma_sum = self.sigma_sum + \
            self.xks @ (self.sigma_sum_factors * self.xks).T
        self.mu_sum = np.zeros_like(self.m)
        self.mu_sum += np.sum((self.Ys - self.sigmoid(mutxks))
                              * self.xks, axis=1, keepdims=True)
        new_d = scipy.linalg.inv(scipy.linalg.inv(self.d0) + self.sigma_sum)
        # if m0 not 0 uncomment
        # new_m = new_d @ scipy.linalg.inv(self.d0) @ self.m0 + self.m + new_d @ self.mu_sum
        new_m = new_d @ (self.sigma_sum @ self.m + self.mu_sum)
        return new_m, new_d

    def update_all_observations_eta(self, k, Y, eta):
        """Calculate current posterior based on all the observations
        including the weighting scheme induced by random walk of the parameters
        """
        xk = self.sk_vector(k)
        if self.xks is None:
            self.xks = xk
        else:
            self.xks = np.hstack((self.xks, xk))
        if self.Ys is None:
            self.Ys = np.array([Y])
        else:
            self.Ys = np.append(self.Ys, Y)
        self.t += 1
        if self.t < 50:
            self.m = self.m0 * 0

        ks = np.linspace(self.t-1, 0, self.t)
        weights = 1/np.sqrt(1+self.f_lambda*ks*eta *
                            np.sum(np.abs(self.xks), axis=0))
        wmutxks = weights * (self.m.reshape(-1) @ self.xks)

        self.d0k = self.d0 + self.t*eta*np.identity(self.dimension)
        self.sigma_sum = np.zeros_like(self.d0)
        self.sigma_sum_factors = self.sigmoid(wmutxks)*(
            1-self.sigmoid(wmutxks))*(weights**2)
        self.sigma_sum = self.sigma_sum + \
            self.xks @ (self.sigma_sum_factors * self.xks).T
        self.mu_sum = np.zeros_like(self.m)
        self.mu_sum += np.sum((self.Ys - self.sigmoid(wmutxks)) * weights
                              * self.xks, axis=1, keepdims=True)
        new_d = scipy.linalg.inv(scipy.linalg.inv(self.d0k) + self.sigma_sum)
        new_m = new_d @ (self.sigma_sum @ self.m + self.mu_sum)
        return new_m, new_d


if __name__ == "__main__":
    a = ARCPricingModel(
        np.array([0., 0.5, 1., 1.5, 2.]), 1., 0.99, dimension=2)

    print(a.get_action())
