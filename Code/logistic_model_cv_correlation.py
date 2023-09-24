"""This script evaluate the correlation between the parameters
of the logistic model and the mean and variance of the observations"""
import numpy as np

from scipy.special import expit


# Initialize the mean and variance of logistic model parameters
mu = np.array([[0.], [0.]])
variance = np.array([[1000., 0.], [0., 1000.]])


def sk_vector(k):
    """Return a vector with entries [1, s, s^2, ...] for given sk"""
    actions = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    dimension = 2
    sk = actions[k]
    sk_vector = np.ones_like(mu)
    for i in range(1, dimension):
        sk_vector[i, 0] = sk_vector[i-1, 0]*sk
    return sk_vector


xks = sk_vector(0)

# Define a function which generates monte carlo samples


def generate_monte_carlo_samples(mu, variance, no_samples):
    """Generate monte carlo samples from a multivariate normal
    distribution with given mean and variance"""
    samples = np.random.multivariate_normal(mu.flatten(),
                                            variance,
                                            size=no_samples)
    return samples


def compute_correlation_mean(samples, k):
    """Compute covariance between the mean of mutxk and the mean of observations
    as well as variance of mutxk"""
    xk = sk_vector(k)
    mutxks = samples.dot(xk)
    probabilities = expit(mutxks)
    covariance = np.cov(
        np.vstack((mutxks.flatten(), probabilities.flatten())), bias=False)
    print(f'Action {k}, optimal c for mean', -
          covariance[0, 1]/covariance[0, 0])
    print(np.corrcoef(mutxks.flatten(), probabilities.flatten()))


def compute_correlation_variance(samples, k):
    """Compute covariance between the mean of mutxk and the variance of observations
    as well as variance of mutxk"""
    xk = sk_vector(k)
    mutxks = samples.dot(xk)
    probabilities = expit(mutxks)
    variances = probabilities*(1-probabilities)
    covariance = np.cov(
        np.vstack((mutxks.flatten(), variances.flatten())), bias=False)
    print(f'Action {k}, optimal c for variance', -
          covariance[0, 1]/covariance[0, 0])
    print(np.corrcoef(mutxks.flatten(), variances.flatten()))


if __name__ == "__main__":

    # mu = np.array([[7.], [-9.5]])
    # variance = np.array([[1.98973408, -2.69456023],
    #                      [-2.69456023,  3.84352896]])
    variance = np.array([[10, 0],
                         [0,  10]])
    samples = generate_monte_carlo_samples(mu, variance, 10000)

    for k in range(5):
        compute_correlation_mean(samples, k)

    for k in range(5):
        compute_correlation_variance(samples, k)
