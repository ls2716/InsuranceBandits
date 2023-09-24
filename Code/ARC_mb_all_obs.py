"""This file contains the implementation of bandit algortihms
for the market problem.
"""
import numpy as np
import random
from scipy.stats import norm
from scipy.special import softmax, expit
import scipy
# Implement eps-greedy bandit


class ARCPricingModel(object):
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

    def setup_prior(self, variance=10**3):
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

    def _f(self):
        """expected rewards f function according to 3.2"""
        if self.f is not None:
            return self.f
        self.mu_prime = self.m[0, 0] + self.actions*self.m[1, 0]
        self.sigma_prime = self.d[0, 0] + 2*self.d[0, 1] * \
            self.actions + self.d[1, 1]*np.power(self.actions, 2)
        f = self.actions * \
            norm.cdf(self.mu_prime /
                     np.sqrt(np.power(self.f_lambda, -2) + self.sigma_prime))
        self.f = f
        return self.f

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

    def _eta(self, a):
        """eta function"""
        nu = self._nu(a)
        eta = -nu @ nu.T + np.diag(nu.flatten())
        return eta

    def reset(self):
        """Reset ARC algorithm"""
        self.setup_prior()

    def dd_fk(self, k):
        """Compute derivative of kth component of f with respect to d"""
        sk = self.actions[k, 0]
        sk_v = self.sk_vector(k)
        dd_fk = - sk*self.mu_prime[k, 0]/2/np.power(np.sqrt(np.power(self.f_lambda, -2) + self.sigma_prime[k, 0]), 3) \
            * norm.pdf(self.mu_prime[k, 0]/np.sqrt(np.power(self.f_lambda, -2) + self.sigma_prime[k, 0])) \
            * sk_v @ sk_v.T
        return dd_fk

    def dm_fk(self, k):
        """Compute derivative of kth component of f with respect to d"""
        sk = self.actions[k, 0]
        sk_v = self.sk_vector(k)
        dm_fk = sk/np.sqrt(np.power(self.f_lambda, -2) + self.sigma_prime[k, 0]) \
            * norm.pdf(self.mu_prime[k, 0]/np.sqrt(np.power(self.f_lambda, -2) + self.sigma_prime[k, 0])) \
            * sk_v
        return dm_fk

    def sk_vector(self, k):
        """Return a vector with entries [1, s, s^2, ...] for given sk"""
        sk = self.actions[k, 0]
        sk_vector = np.ones_like(self.m)
        for i in range(1, self.dimension):
            sk_vector[i, 0] = sk_vector[i-1, 0]*sk
        return sk_vector

    def phi_prime(self, x):
        """Return phi prime"""
        return -x/np.sqrt(2*np.pi)*np.exp(-x*x/2)

    def dm2_fk(self, k):
        """Compute derivative of kth component of f with respect to d"""
        sk = self.actions[k, 0]
        sk_v = self.sk_vector(k)
        dm2_fk = sk/(np.power(self.f_lambda, -2) + self.sigma_prime[k, 0]) \
            * self.phi_prime(self.mu_prime[k, 0]/np.sqrt(np.power(self.f_lambda, -2) + self.sigma_prime[k, 0])) \
            * sk_v @ sk_v.T
        return dm2_fk

    def _B(self, nu):
        """Calculate B matrix for the learning function"""
        B = np.zeros_like(self.d)
        for k in range(self.K):
            B += nu[k, 0] * self.dd_fk(k)
        return B

    def _M(self, nu):
        """Calculate M matrix for the learning function"""
        M = np.zeros_like(self.m)
        for k in range(self.K):
            M += nu[k, 0] * self.dm_fk(k)
        return M

    def _Xi(self, nu, eta):
        """Calculate Xi matrix for the learning function"""
        Xi = np.zeros_like(self.d)
        for k in range(self.K):
            Xi += nu[k, 0] * self.dm2_fk(k)
        Xi_cross = np.zeros_like(self.d)
        for k in range(self.K):
            for j in range(self.K):
                dm_fk = self.dm_fk(k)
                Xi_cross += eta[j, k] * (dm_fk @ dm_fk.T)
        Xi = Xi + Xi_cross/self.lamda
        return Xi

    def sigmoid(self, x):
        """Sigmoid function"""
        return expit(x)

    def generate_monte_carlo_sample(self):
        """Generate mc sample for monte carlo integration"""
        self.mc_sample = np.random.multivariate_normal(
            self.m.flatten(), self.d, self.mc_size)

    def E_Yk(self, k):
        """Calculate expectation of Yk given current posterior"""
        xk = self.sk_vector(k)
        mutxk_sample = (self.mc_sample @ xk)
        probs_sample = self.sigmoid(mutxk_sample)
        c = -0.1
        return np.mean(probs_sample + c*(mutxk_sample - self.m.T @ xk))

    def Var_Yk(self, k):
        """Calculate expectation of Yk given current posterior"""
        xk = self.sk_vector(k)
        mutxk_sample = (self.mc_sample @ xk)
        probs_sample = self.sigmoid(mutxk_sample)
        vars_sample = probs_sample * (1-probs_sample)
        return np.mean(vars_sample)

    def Var_mk(self, k):
        """Calculate variance of change in m if k-th arm
        is chosen.
        """
        xk = self.sk_vector(k)
        dtp1 = self.dtp1(k)
        sigxk = dtp1 @ xk
        varmk = sigxk @ sigxk.T * self.Var_Yk(k)
        return varmk

    def E_mk(self, k):
        """Calculate expected change in m if k-th arm
        is chosen.
        """
        if self.t < 50:
            self.m = self.m0 * 0
        xk = self.sk_vector(k)
        if self.xks is None:
            self.xks_p = xk
        else:
            self.xks_p = np.hstack((self.xks, xk))
        mutxks = self.m.reshape(-1) @ self.xks_p
        self.sigma_sum = np.zeros_like(self.d0)
        self.sigma_sum_factors = self.sigmoid(mutxks)*(
            1-self.sigmoid(mutxks))
        self.sigma_sum = self.sigma_sum + \
            self.xks_p @ (self.sigma_sum_factors * self.xks_p).T
        dtp1 = scipy.linalg.inv(scipy.linalg.inv(self.d0) + self.sigma_sum)

        mtp1 = (scipy.linalg.inv(self.d0) @ self.m0 + self.sigma_sum @ self.m)
        if self.xks is not None:
            mutxks = self.m.reshape(-1) @ self.xks
            mtp1 += np.sum((self.Ys - self.sigmoid(mutxks))
                           * self.xks, axis=1, keepdims=True)
        mtp1 = mtp1 - self.sigmoid(self.m.T @ xk) * \
            xk - self.m + self.E_Yk(k) * xk
        Emk = dtp1 @ mtp1
        return Emk

    def E_dk(self, k):
        """Calculate expected change in d if k-th arm
        is chosen.
        """
        if self.t < 50:
            self.m = self.m0 * 0
        xk = self.sk_vector(k)
        if self.xks is None:
            self.xks_p = xk
        else:
            self.xks_p = np.hstack((self.xks, xk))
        mutxks = self.m.reshape(-1) @ self.xks_p
        self.sigma_sum = np.zeros_like(self.d0)
        self.sigma_sum_factors = self.sigmoid(mutxks)*(
            1-self.sigmoid(mutxks))
        self.sigma_sum = self.sigma_sum + \
            self.xks_p @ (self.sigma_sum_factors * self.xks_p).T
        dtp1 = scipy.linalg.inv(scipy.linalg.inv(self.d0) + self.sigma_sum)
        return dtp1 - self.d

    def dtp1(self, k):
        """Compute new sigma"""
        dtp1 = self.d + self.E_dk(k)
        return dtp1

    def learning_function(self):
        """Evaluate the learning function
        """
        f = self._f()
        nu = self._nu(f)
        eta = self._eta(f)
        Xi = self._Xi(nu, eta)
        M = self._M(nu)
        B = self._B(nu)
        self.generate_monte_carlo_sample()
        self.L = np.zeros_like(self.actions)
        for k in range(self.K):
            self.L[k] = np.dot(M.flatten(), self.E_mk(k).flatten()) +\
                np.trace(B.T @ self.E_dk(k)) + \
                np.trace(Xi.T @ self.Var_mk(k))
        return self.L

    def get_probability(self):
        """Calculate probability using (a) variant

        That is, we use f to evaluate learning function, instead of
        finding the fixed point of the equation.
        """
        L = self.learning_function()
        self.U = self._nu(self.f + self.beta/(1-self.beta)*L)
        # self.U = self._nu(self._f())  # Without learning function
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
        if self.t < 50:
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

    # def update_all_observations_eta(self, k, Y, eta):
    #     """Calculate current posterior based on all the observations
    #     including the weighting scheme induced by random walk of the parameters
    #     """
    #     xk = self.sk_vector(k)
    #     if self.xks is None:
    #         self.xks = xk
    #     else:
    #         self.xks = np.hstack((self.xks, xk))
    #     if self.Ys is None:
    #         self.Ys = np.array([Y])
    #     else:
    #         self.Ys = np.append(self.Ys, Y)
    #     self.t += 1
    #     if self.t < 10:
    #         self.m = self.m0 * 0

    #     ks = np.linspace(self.t-1, 0, self.t)
    #     weights = 1/np.sqrt(1+self.f_lambda*ks*eta *
    #                         np.sum(np.abs(self.xks), axis=0))
    #     wmutxks = weights * (self.m.reshape(-1) @ self.xks)

    #     self.d0k = self.d0 + self.t*eta*np.identity(self.dimension)
    #     self.sigma_sum = np.zeros_like(self.d0)
    #     self.sigma_sum_factors = self.sigmoid(wmutxks)*(
    #         1-self.sigmoid(wmutxks))*(weights**2)
    #     # print("sigma sum factor shape", self.sigma_sum_factors.shape)
    #     # print(self.sigma_sum_factors)
    #     # print((self.sigma_sum_factors * self.xks).T)
    #     self.sigma_sum = self.sigma_sum + \
    #         self.xks @ (self.sigma_sum_factors * self.xks).T
    #     # print("sigma sum shape", self.sigma_sum.shape)
    #     self.mu_sum = np.zeros_like(self.m)
    #     # print((self.Ys - self.sigmoid(mutxks)))
    #     # print(np.sum((self.Ys - self.sigmoid(mutxks))
    #     #              * self.xks, axis=1, keepdims=True))
    #     self.mu_sum += np.sum((self.Ys - self.sigmoid(wmutxks)) * weights
    #                           * self.xks, axis=1, keepdims=True)
    #     # print('mu sum', self.mu_sum.shape)
    #     new_d = scipy.linalg.inv(scipy.linalg.inv(self.d0k) + self.sigma_sum)
    #     new_m = new_d @ (self.sigma_sum @ self.m + self.mu_sum)
    #     return new_m, new_d


if __name__ == "__main__":
    a = ARCPricingModel(
        np.array([0., 0.5, 1., 1.5, 2.]), 1., 0.99, dimension=2)
    a.generate_monte_carlo_sample()
    a._f()
    a.learning_function()
    # print(a.L)

    # print(a.sk_vector(1) @ a.sk_vector(1).T)
    a.update_all_observations(0, 1.)
    print(a.update_all_observations(1, 0.))
