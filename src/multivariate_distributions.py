import numpy as np
import scipy
from scipy.stats import norm
from scipy.stats._multivariate import multi_rv_generic, multi_rv_frozen, multivariate_normal, multivariate_normal_gen, _squeeze_output, binom
from src.stats import *

class simple_multivariate_normal_mixture_gen(multivariate_normal_gen):
    """
    A class representing a simple multivariate normal mixture distribution.
    
    This class provides methods to compute the PDF, random variate samples, and other common functions
    for a simple multivariate normal mixture distribution composed of two normal distributions.

    Methods
    -------
    __call__(mu1=None, cov1=1, mu2=None, cov2=1, p=0.5, allow_singular=False, seed=None)
        Create a frozen multivariate normal mixture distribution.
    pdf(x, mu1=None, cov1=1, mu2=None, cov2=1, p=0.5, allow_singular=False)
        Probability density function.
    rvs(mu1=None, cov1=1, mu2=None, cov2=1, p=0.5, size=1, random_state=None)
        Draw random samples from the distribution.
    """
    def __init__(self, seed=None):
        super().__init__(seed)

    def __call__(self, mu1=None, cov1=1, mu2=None, cov2=1, p=0.5, allow_singular=False, seed=None):
        """
        Creates a frozen multivariate normal mixture distribution.

        Parameters
        ----------
        mu1 : array_like, optional
            Mean of the first normal distribution.
        cov1 : array_like, optional
            Covariance matrix of the first normal distribution.
        mu2 : array_like, optional
            Mean of the second normal distribution.
        cov2 : array_like, optional
            Covariance matrix of the second normal distribution.
        p : float, optional
            Weight of the first normal distribution in the mixture.
        allow_singular : bool, optional
            Whether to allow a singular covariance matrix. (Not recommended.)
        seed : int or None, optional
            Seed for the random number generator.

        Returns
        -------
        dist : simple_multivariate_normal_mixture_frozen
            A frozen multivariate normal mixture distribution.
        """
        return simple_multivariate_normal_mixture_frozen(mu1, cov1, mu2, cov2, p, allow_singular, seed)

    def _process_parameters(self, mu1, cov1, mu2, cov2, p, allow_singular=True):
        """
        Internal method to process the parameters of the multivariate normal distributions.

        Parameters are checked for consistency and reshaped if necessary.

        Parameters
        ----------
        mu1, mu2 : array_like
            Means of the multivariate normal distributions.
        cov1, cov2 : array_like
            Covariance matrices of the multivariate normal distributions.
        p : float
            Mixture coefficient for the distributions.
        allow_singular : bool
            Whether to allow singular covariance matrices.

        Returns
        -------
        dim : int
            Dimensionality of the distributions.
        mu1, mu2 : ndarray
            Processed means.
        cov1, cov2 : ndarray
            Processed covariance matrices.
        p : float
            Mixture coefficient.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        dim, mu1, cov1 = super()._process_parameters(mu1, cov1, allow_singular)
        _, mu2, cov2 = super()._process_parameters(mu2, cov2, allow_singular)
        if not np.isscalar(p):
            raise ValueError("Parameter p must be a scalar.")
        if p > 1 or p < 0:
            raise ValueError("Parameter p must be in [0, 1]")
        return dim, mu1, cov1, mu2, cov2, p

    def _pdf(self, x, mu1, cov1, mu2, cov2, p):
        """
        Computes the probability density function (PDF) for the mixture distribution at the given points.

        The PDF for a mixture of two multivariate normal distributions is given by:

        f(x) = p * N(x | mu1, cov1) + (1 - p) * N(x | mu2, cov2)

        where N(x | mu, cov) is the PDF of a multivariate normal distribution with mean mu and covariance cov.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the PDF.
        mu1, mu2 : array_like
            Means of the component distributions.
        cov1, cov2 : array_like
            Covariance matrices of the component distributions.
        p : float
            Mixture coefficient for the first component distribution.

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `x`.
        """
        return p * multivariate_normal.pdf(x, mu1, cov1) + (1 - p) * multivariate_normal.pdf(x, mu2, cov2)

    def pdf(self, x, mu1=None, cov1=1, mu2=None, cov2=1, p=0.5, allow_singular=False):
        """
        Probability density function of the multivariate normal mixture distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the PDF.
        mu1 : array_like, optional
            Mean of the first normal distribution.
        cov1 : array_like, optional
            Covariance matrix of the first normal distribution.
        mu2 : array_like, optional
            Mean of the second normal distribution.
        cov2 : array_like, optional
            Covariance matrix of the second normal distribution.
        p : float, optional
            Weight of the first normal distribution in the mixture.
        allow_singular : bool, optional
            Whether to allow a singular covariance matrix. (Not recommended.)

        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at `x`.
        """
        dim, mu1, cov1, mu2, cov2, p = self._process_parameters(mu1, cov1, mu2, cov2, p, allow_singular)
        x = self._process_quantiles(x, dim)
        out = self._pdf(x, mu1, cov1, mu2, cov2, p)
        return _squeeze_output(out)

    def _rvs(self, mu1, cov1, mu2, cov2, p, size, random_state):
        """
        Generate random variates of size `size`.

        Parameters
        ----------
        mu1, mu2 : array_like
            Means of the component distributions.
        cov1, cov2 : array_like
            Covariance matrices of the component distributions.
        p : float
            Mixture coefficient for the first component distribution.
        size : int
            Number of random variates to generate.
        random_state : RandomState or None
            Random number generator or None to use the global `numpy.random`.

        Returns
        -------
        rvs : ndarray
            Random variates of the mixture distribution.
        """
        num_samples1 = np.random.binomial(n=size, p=p)
        num_samples2 = size - num_samples1
        samples1 = multivariate_normal.rvs(mu1, cov1, num_samples1, random_state)
        samples2 = multivariate_normal.rvs(mu2, cov2, num_samples2, random_state)
        x = np.concatenate((samples1, samples2))
        random_state.shuffle(x)
        return x

    def rvs(self, mu1=None, cov1=1, mu2=None, cov2=1, p=0.5, size=1, random_state=None):
        """
        Draw random variates from a multivariate normal mixture distribution.

        Parameters
        ----------
        mu1 : array_like, optional
            Mean of the first normal distribution.
        cov1 : array_like, optional
            Covariance matrix of the first normal distribution.
        mu2 : array_like, optional
            Mean of the second normal distribution.
        cov2 : array_like, optional
            Covariance matrix of the second normal distribution.
        p : float, optional
            Weight of the first normal distribution in the mixture.
        size : int, optional
            Number of random variates to generate.
        random_state : RandomState or None, optional
            Random number generator or None to use the global `numpy.random`.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of the mixture distribution.
        """
        dim, mu1, cov1, mu2, cov2, p = self._process_parameters(mu1, cov1, mu2, cov2, p, allow_singular=False)
        random_state = self._get_random_state(random_state)
        out = self._rvs(mu1, cov1, mu2, cov2, p, size, random_state)
        return _squeeze_output(out)

simple_multivariate_normal_mixture = simple_multivariate_normal_mixture_gen()

class simple_multivariate_normal_mixture_frozen(multi_rv_frozen):
    def __init__(self, mu1=None, cov1=1, mu2=None, cov2=1, p=0.5, allow_singular=False, seed=None):
        """
        Frozen multivariate normal mixture distribution.

        Parameters
        ----------
        mu1 : array_like, optional
            Mean of the first normal distribution.
        cov1 : array_like, optional
            Covariance matrix of the first normal distribution.
        mu2 : array_like, optional
            Mean of the second normal distribution.
        cov2 : array_like, optional
            Covariance matrix of the second normal distribution.
        p : float, optional
            Weight of the first normal distribution in the mixture.
        allow_singular : bool, optional
            Whether to allow a singular covariance matrix. (Not recommended.)
        seed : int or None, optional
            Seed for the random number generator.

        Methods
        -------
        pdf(x)
            Probability density function.
        rvs(size=1, random_state=None)
            Draw random samples from the distribution.
        """
        self._dist = simple_multivariate_normal_mixture_gen(seed)
        self.dim, self.mu1, self.cov1, self.mu2, self.cov2, self.p = self._dist._process_parameters(mu1, cov1, mu2, cov2, p, allow_singular)

    def pdf(self, x):
        """
        Probability density function at x of the given RV.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the PDF.

        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at `x`.
        """
        return self._dist.pdf(x, self.mu1, self.cov1, self.mu2, self.cov2, self.p)

    def rvs(self, size=1, random_state=None):
        """
        Draw random samples from the distribution.

        Parameters
        ----------
        size : int, optional
            Number of random variates to generate.
        random_state : RandomState or None, optional
            Random number generator or None to use the global `numpy.random`.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of the mixture distribution.
        """
        return self._dist.rvs(self.mu1, self.cov1, self.mu2, self.cov2, self.p, size, random_state)
    
