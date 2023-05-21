import numpy as np
import scipy
from scipy.stats import powerlaw, norm, t, cauchy, pareto, gamma, lognorm, rv_continuous
from src.stats import *

class double_pareto_gen(rv_continuous):
    """
    Double (sided) Pareto distribution

    Parameters
    ----------
    alpha : float
        The shape parameter.
    """
    def _argcheck(self, alpha):
        return alpha > 0

    def _pdf(self, x, alpha):
        p = np.zeros(len(x))
        if isinstance(alpha, int) or isinstance(alpha, float):
            p[x >= 0] = pareto.pdf(1 + x[x >= 0], b=alpha) / 2
            p[x < 0] = pareto.pdf(1 - x[x < 0], b=alpha) / 2
        else:
            p[x >= 0] = pareto.pdf(1 + x[x >= 0], b=alpha[x >= 0]) / 2
            p[x < 0] = pareto.pdf(1 - x[x < 0], b=alpha[x < 0]) / 2
        return p

    def _cdf(self, x, alpha):
        c = np.zeros(len(x))
        c[x > 0] = pareto.cdf(1 + x[x > 0], b=alpha) / 2 + 0.5
        c[x < 0] = 0.5 - pareto.cdf(1 - x[x < 0], b=alpha) / 2
        return c

    def _ppf(self, q, alpha):
        x = np.zeros(len(q))
        x[q >= 0.5] = pareto.ppf(2 * (q[q >= 0.5] - 0.5), b=alpha) - 1
        x[q < 0.5] = 1 - pareto.ppf(2 * (0.5 - q[q < 0.5]), b=alpha)
        return x

    def _stats(self, alpha):
        mean = 0
        var = pareto.var(b=alpha) + (pareto.mean(b=alpha) - 1) ** 2
        return mean, var, None, None


class hypsecant2_gen(rv_continuous):
    """
    Hypsecant2 is a class representing the distribution with pdf
    $$
    f(x) = \frac{1}{2} sech^2(x)
    $$
    which is the square of the hyper secant distribution.

    Parameters
    ----------
    k : float
        The parameter for the distribution.
    """
    def _argcheck(self):
        return True

    def _pdf(self, x):
        return 0.5 / np.cosh(x)**2

    def _cdf(self, x):
        return 0.5 * (1 + np.tanh(x))

    def _ppf(self, q):
        return np.arctanh(2 * q - 1)
    
    def chf(self, x, loc=0, scale=1):
        return (np.pi * x * csch(np.pi * x * scale / 2)) * scale / 2
    

class simple_normal_mixture_gen(rv_continuous):
    """
    A mixture of two Gaussian distributions with the same mean but different variances.
    pdf: 
    $$
    f(x) = \frac{1}{2} \phi(\frac{x}{1+a}) + \frac{1}{2} \phi(\frac{x}{1-a})
    $$
    where $\phi$ is the standard normal pdf.
    
    Attributes:
        a (float): Tail parameter.
        loc (float): Location parameter, same as m.
        scale (float): Scale parameter, same as s.        
    """
    def _argcheck(self, a):
        return (a <= 1) & (a >= 0)
    
    def _pdf(self, x, a):
        s1 = 1 + a 
        s2 = 1 - a
        return 0.5 * (norm.pdf(x, 0, np.sqrt(s1)) + norm.pdf(x, 0, np.sqrt(s2)))
    
    def _cdf(self, x, a):
        s1 = 1 + a 
        s2 = 1 - a
        return 0.5 * (norm.cdf(x, 0, np.sqrt(s1)) + norm.cdf(x, 0, np.sqrt(s2)))

    def _stats(self, a):
        mean = 0
        var = 1
        return mean, var, None, None

    def _rvs(self, a, size=None, random_state=None):
        s1 = 1 + a 
        s2 = 1 - a
        choices = random_state.choice([s1, s2], size=size, p=[0.5, 0.5])
        return norm.rvs(loc=0, scale=np.sqrt(choices), random_state=random_state)
    
class univariate_variance_gamma_gen(rv_continuous):
    """
    A univariate Variance Gamma distribution with pdf:
    $$
    f(x) = \frac{{(\alpha^2 - \beta^2)^{\lambda} |x|^{\lambda - 0.5} K_{\lambda - 0.5}(\alpha |x|)}}{{\sqrt{\pi} \Gamma(\lambda) (2\alpha)^{\lambda - 0.5}}} e^{\beta x}
    $$
    where:
    - \(\alpha\) is the scale parameter.
    - \(\beta\) is the skewness parameter.
    - \(\lambda\) is the shape parameter.
    - \(K_{\lambda - 0.5}(\cdot)\) is the modified Bessel function of the second kind.
    - \(\Gamma(\cdot)\) is the gamma function.

    Parameters
    ----------
    alpha : float
        Scale parameter.
    beta : float
        Skewness parameter.
    lam : float
        Shape parameter.
    """
    def _argcheck(self, alpha, beta, lam):
        return (alpha > beta) & (lam > 0)

    def _pdf(self, x, alpha, beta, lam):
        numerator = ((alpha**2 - beta**2) ** lam) * (np.abs(x) ** (lam - 0.5)) * scipy.special.kv(lam - 0.5, alpha * np.abs(x))
        denominator = (np.sqrt(np.pi) * scipy.special.gamma(lam) * (2 * alpha) ** (lam - 0.5))
        return (numerator / denominator) * np.exp(beta * x)

    def _rvs(self, alpha, beta, lam, size=None, random_state=None):
        theta = 2/(alpha**2 - beta**2)
        gamma_rvs = gamma.rvs(lam, loc=0, scale=theta, size=size, random_state=random_state)
        norm_rvs = norm.rvs(loc=0, scale=1, size=size, random_state=random_state)
        return beta*gamma_rvs + np.sqrt(gamma_rvs)*norm_rvs

    def _stats(self, alpha, beta, lam):
        theta = 2/(alpha**2 - beta**2)
        mean = beta*lam*theta
        var = lam*theta * (1 + beta**2 * theta)
        return mean, var, None, None

double_pareto = double_pareto_gen(name='double_pareto')
hypsecant2 = hypsecant2_gen(name='hypsecant2')
simple_normal_mixture = simple_normal_mixture_gen(name='simple_normal_mixture')
univariate_variance_gamma = univariate_variance_gamma_gen(name='univariate_variance_gamma')

