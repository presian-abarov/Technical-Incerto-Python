import numpy as np
import scipy
from scipy.stats import powerlaw, norm, t, cauchy, pareto, gamma, lognorm

class SimpleNormalMixture:
    """
    A mixture of two Gaussian distributions with the same mean but different variances.
    
    PDF formula: 1/2 * f(x, m, (1+a)s) + 1/2 * f(x, m, (1-a)s),
    where f(x, m, s) is the normal PDF with mean m and variance s.
    
    Attributes:
        m (float): The mean of both Gaussian distributions.
        s (float): The base variance used to calculate the variances of the two Gaussian distributions.
        a (float): The factor used to adjust the variance for each Gaussian distribution.
    """
    def __init__(self, m, s, a):
        self.m = m
        self.s = s
        self.a = a

    def pdf(self, x):
        s1 = (1 + self.a) * self.s
        s2 = (1 - self.a) * self.s
        return 0.5 * (norm.pdf(x, self.m, np.sqrt(s1)) + norm.pdf(x, self.m, np.sqrt(s2)))

    def cdf(self, x):
        s1 = (1 + self.a) * self.s
        s2 = (1 - self.a) * self.s
        return 0.5 * (norm.cdf(x, self.m, np.sqrt(s1)) + norm.cdf(x, self.m, np.sqrt(s2)))

    def rvs(self, size=None):
        s1 = (1 + self.a) * self.s
        s2 = (1 - self.a) * self.s
        choices = np.random.choice([s1, s2], size=size, p=[0.5, 0.5])
        return norm.rvs(loc=self.m, scale=np.sqrt(choices))

    def mean(self):
        return self.m

    def var(self):
        return self.s
    
    def std(self):
        return np.sqrt(self.s)


class UnivariateVarianceGamma:
    """
    A univariate Variance Gamma distribution.
    
    Attributes:
        mu (float): Location parameter.
        alpha (float): Scale parameter.
        beta (float): Skewness parameter.
        lam (float): Shape parameter.
    """
    def __init__(self, mu, alpha, beta, lam):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.lam = lam

    def pdf(self, x):
        numerator = (self.alpha ** (2 * self.lam)) * (np.abs(x - self.mu) ** (self.lam - 0.5)) * scipy.special.kv(self.lam - 0.5, self.alpha * np.abs(x - self.mu))
        denominator = (np.sqrt(np.pi) * scipy.special.gamma(self.lam) * (2 * self.alpha) ** (self.lam - 0.5))
        return (numerator / denominator) * np.exp(self.beta * (x - self.mu))
    
    def rvs(self, size):
        theta = 2/(self.alpha**2 - self.beta**2)
        gamma_rvs = gamma.rvs(self.lam, loc=0, scale=theta, size=size)
        norm_rvs = norm.rvs(loc=0, scale=1, size=size)
        return self.mu + self.beta*gamma_rvs + np.sqrt(gamma_rvs)*norm_rvs
    
    def var(self):
        theta = 2/(self.alpha**2 - self.beta**2)
        return self.lam*theta * (1 + self.beta**2 * theta)
    
    def mean(self):
        theta = 2/(self.alpha**2 - self.beta**2)
        return self.mu + self.beta*self.lam*theta