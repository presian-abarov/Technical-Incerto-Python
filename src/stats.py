import numpy as np
import pandas as pd
from scipy.fft import fft, ifft, fftshift
from scipy.stats import powerlaw, norm, t, cauchy, pareto
from sklearn.linear_model import LinearRegression, HuberRegressor

def chf_to_pdf(chf, x_min=-10, x_max=10, n_points=1000):
    """
    Computes the PDF numerically from a given characteristic function using Fourier transformation.

    Args:
        char_func (function): Characteristic function of the distribution.
        x_min (float, optional): Minimum value of x. Default is -10.
        x_max (float, optional): Maximum value of x. Default is 10.
        n_points (int, optional): Number of points to use in the discretization. Default is 1000.

    Returns:
        tuple: A tuple containing the x values (np.array) and the corresponding PDF values (np.array).
    """

    # Define the grid for x values
    x = np.linspace(x_min, x_max, n_points)

    # Define the grid for frequency values
    freq = np.fft.fftfreq(n_points, d=x[1]-x[0]) * 2 * np.pi

    # Evaluate the characteristic function on the frequency grid
    chf_vals = chf(freq)
    chf_vals[0] = 1

    # Compute the inverse Fourier transform of the characteristic function values
    pdf_vals = np.real(fftshift(ifft(chf_vals)))

    # Normalize the PDF
    pdf_vals /= (x[1] - x[0])

    return x, pdf_vals

def csch(x):
    """
    Compute the hyperbolic cosecant (csch) of x.
    
    Args:
        x (float): The input value.
        
    Returns:
        float: The hyperbolic cosecant of x.
    """
    return 1 / np.sinh(x)

def detect_simple_regression_outliers(x, y, threshold=2.0):
    """
    Detect outliers of a simple linear regression model using robust regression and z-scores.

    Args:
        x (array_like): Predictor variable.
        y (array_like): Response variable.
        threshold (float, optional): Threshold for z-scores. Observations with z-scores greater than this
                                      threshold are considered outliers. Default is 2.0.

    Returns:
        outliers (ndarray): Indices of outlier observations.

    """
    # Fit a robust linear regression model using HuberRegressor
    model = HuberRegressor()
    model.fit(x.reshape(-1, 1), y)
    
    # Compute the residuals of the robust regression
    residuals = y - model.predict(x.reshape(-1, 1)).flatten()
    
    # Compute the median absolute deviation of the residuals
    mad = median_absolute_deviation(residuals)
    
    # Compute the z-scores of the residuals
    z_scores = np.abs(residuals / mad)
    
    # Identify the outliers based on the z-scores
    outliers = np.where(z_scores > threshold)[0]
    
    return outliers


def median_absolute_deviation(x, axis=None):
    """
    Compute the median absolute deviation of an array x, adjusted to be 1 for a standard normal distribution.
    
    Args:
        x (array_like): Input data.
        axis (int, optional): Axis along which to compute the median absolute deviation. 
                     If None, compute the MAD for the flattened array.
        
    Returns:
        float: Median absolute deviation of x, adjusted for a standard normal distribution.
    """
    # Compute the median of x
    if axis is None:
        med = np.median(x)
    else:
        med = np.median(x, axis=axis, keepdims=True)
    
    # Compute the median absolute deviation of x, and adjust for standard normal distribution
    mad = np.median(np.abs(x - med), axis=axis) / 0.6745
    
    return mad

def mean_absolute_deviation(x, axis=None):
    """
    Compute the mean absolute deviation of an array x, adjusted to be 1 for a standard normal distribution.
    
    Args:
        x (array_like): Input data.
        axis (int, optional): Axis along which to compute the mean absolute deviation. 
                             If None, compute the MAD for the flattened array.
        
    Returns:
        float or ndarray: Mean absolute deviation of x, adjusted for a standard normal distribution.
    """
    # Compute the mean of x
    if axis is None:
        mu = np.mean(x)
    else:
        mu = np.mean(x, axis=axis, keepdims=True)
    
    # Compute the mean absolute deviation of x, and adjust for standard normal distribution
    mad = np.mean(np.abs(x - mu), axis=axis) * np.sqrt(np.pi/2)
    
    return mad

def approximate_relative_efficiency_ratio(x, denominator=mean_absolute_deviation, numerator=np.std):
    """
    Compute the approximate relative efficiency ratio of two estimators (numerator and denominator).
    
    Args:
        x (array_like): Input data, a 2D array with shape (m, n).
        denominator (callable): Function to compute the denominator's estimator, default is mean_absolute_deviation.
        numerator (callable): Function to compute the numerator's estimator, default is np.std.

    Returns:
        float: Approximate relative efficiency ratio of the numerator's estimator to the denominator's estimator.

    """
    m, n = x.shape
    
    num = numerator(x, axis=1)
    num_eff = np.var(num) / np.mean(num)
    
    den = denominator(x, axis=1)
    den_eff = np.var(den) / np.mean(den)
    
    return num_eff/den_eff


def lognorm_params_from_mean_and_variance(mean, variance):
    """
    Calculate the mu and sigma parameters of a log-normal distribution given its mean and variance.
    
    Args:
        mean (float): The mean of the log-normal distribution.
        variance (float): The variance of the log-normal distribution.
    
    Returns:
        tuple: A tuple containing mu (float) and sigma (float) parameters of the log-normal distribution.
    """
    sigma_squared = np.log(variance / (mean ** 2) + 1)
    sigma = np.sqrt(sigma_squared)
    mu = np.log(mean) - 0.5 * sigma_squared

    return mu, sigma


def find_peak_shoulder_tail(pdf, lb=-20, ub=20, mu=0, sigma=1, dx=0.01, delta=0.5):
    """
    Identify and label the peak, shoulders, and tails of a quasi-concave probability distribution function (pdf). 
    Reference: Statistical consequences of fat tails, section 4.3.

    Args:
        pdf (callable): The probability distribution function to analyze, which must be sysmetric and centered at zero.
        lb (float, optional): The lower bound of the x range to consider.
        ub (float, optional): The upper bound of the x range to consider.
        mu (float, optional): The location parameter of the pdf.
        sigma (float, optional): The scale parameter of the pdf.
        dx (float, optional): The step size for discretizing the x range. Default is 0.01.
        delta (float, optional): The scaling factor for comparing the original pdf with scaled pdfs. Default is 0.5.

    Returns:
        pd.DataFrame: A DataFrame containing the x values, original pdf values, scaled pdf values, and labels for
                      the peak, shoulders, and tails of the pdf.

    Raises:
        AssertionError: If the number of changing points found in the pdf is not equal to 4, indicating that
                        the pdf may not be quasi-concave.
    """
    x = np.arange(lb, ub, dx)
    p = pdf((x-mu)/sigma)/sigma
    p_delta = 0.5*(pdf((x-mu)/sigma/(1+delta))/sigma/(1+delta)+pdf((x-mu)/sigma/(1-delta))/sigma/(1-delta))
    
    sign_change = np.where(np.diff(np.sign(p - p_delta)))[0]
    assert len(sign_change)==4, 'Cannot find 4 changing points'
    
    output = pd.DataFrame({'x': x, 'pdf': p, 'pdf_delta': p_delta})
    output.loc[:sign_change[0], 'pdf_body'] = 'left_tail'
    output.loc[sign_change[0]:sign_change[1], 'pdf_body'] = 'left_shoulder'
    output.loc[sign_change[1]:sign_change[2], 'pdf_body'] = 'peak'
    output.loc[sign_change[2]:sign_change[3], 'pdf_body'] = 'right_shoulder'
    output.loc[sign_change[3]:, 'pdf_body'] = 'right_tail'

    return output


def normal_peak_shoulder_tail(mu, sigma):
    """
    Calculate peak, shoulder, and tail boundaries for a normal pdf.
    
    Args:
        mu (float): The mean of the normal distribution.
        sigma (float): The standard deviation of the normal distribution.
        
    Returns:
        tuple: A tuple containing the boundaries (left_tail, left_shoulder, right_shoulder, right_tail).
    """
    x1 = np.sqrt((5 + np.sqrt(17)) / 2)
    x2 = np.sqrt((5 - np.sqrt(17)) / 2)
    return (mu - x1 * sigma,
            mu - x2 * sigma,
            mu + x2 * sigma,
            mu + x1 * sigma)

def t_peak_shoulder_tail(mu, sigma, alpha):
    """
    Calculate peak, shoulder, and tail boundaries for a Student's t pdf.
    
    Args:
        mu (float): The location parameter of the t-distribution.
        sigma (float): The scale parameter of the t-distribution.
        alpha (float): The degrees of freedom of the t-distribution.
        
    Returns:
        tuple: A tuple containing the boundaries (left_tail, left_shoulder, right_shoulder, right_tail).
    """
    x1 = np.sqrt(0.5*(5*alpha + np.sqrt((alpha+1)*(17*alpha+1)) + 1) / (alpha-1))
    x2 = np.sqrt(0.5*(5*alpha - np.sqrt((alpha+1)*(17*alpha+1)) + 1) / (alpha-1))
    return (mu - x1 * sigma,
            mu - x2 * sigma,
            mu + x2 * sigma,
            mu + x1 * sigma)

def rousseeuw_croux_sn(x):
    """
    Compute the Sn scale estimator for the given x, as proposed by Rousseeuw and Croux (1993).
    
    Args:
        x (array-like): A 1D NumPy array or a list of data points.
    
    Returns:
        float: The Sn scale estimator for the input data.
    
    References:
        Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the Median Absolute Deviation.
        Journal of the American Statistical Association, 88(424), 1273-1283.
    """
    x = np.asarray(x)
    n = len(x)
    abs_diffs = np.abs(x[:, None] - x)
    medians = np.median(abs_diffs, axis=1)
    sn = 1.1926 * np.median(medians)
    return sn

def rousseeuw_croux_qn(x):
    """
    Compute the Qn scale estimator for the given x, as proposed by Rousseeuw and Croux (1993),
    with an updated scale coefficient from Akinshin (2022).
    
    Args:
        x (array-like): A 1D NumPy array or a list of data points.
    
    Returns:
        float: The Qn scale estimator for the input data.
    
    References:
        Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the Median Absolute Deviation.
        Journal of the American Statistical Association, 88(424), 1273-1283.
        
        Akinshin, A. (2022). Finite-sample Rousseeuw-Croux scale estimators.
    """
    x = np.asarray(x)
    n = len(x)
    abs_diffs = np.abs(np.subtract.outer(x, x))
    abs_diffs = abs_diffs[np.triu_indices(n, k=1)]
    qn = 2.2191 * np.percentile(abs_diffs, 25)
    return qn

def empirical_survival_function(x):
    """Computes the empirical survival function of the input data.

    Args:
        x (numpy array): Input data.

    Returns:
        tuple: Sorted input data and corresponding survival function values.
    """
    x = np.sort(x)
    counts = np.arange(len(x), 0, -1)
    sf = counts / len(x)
    return x, sf


class PowerLawEmpirical(object):
    """Power law model for survival analysis.

    The survival function is modeled as:
    $$P(X>x) \sim C e^{-\alpha}$$
    $$\log P(X>x) = \log C - \alpha \log(x)$$

    Attributes:
        min_size (int): Minimum size of the input data for the model to be fitted.
    """
    def __init__(self, min_size=100):
        """Initializes the PowerLawEmpirical with the given minimum size.

        Args:
            min_size (int, optional): Minimum size of the input data for the model to be fitted. Defaults to 100.
        """
        self.min_size = min_size
        
    def fit(self, x):
        """Fits the power law model to the input data.

        Args:
            x (numpy array): Input data.

        Returns:
            PowerLawEmpirical: The fitted model.
        """
        x, sf = empirical_survival_function(x[x>0])
        self.x = x
        self.sf = sf
        
        n = len(x)
        logx = np.log(x)
        logsf = np.log(sf)
        
        self.r2_adj = np.zeros(n-self.min_size)
        for i in range(n-self.min_size):
            lr = LinearRegression()
            lr.fit(logx[i:].reshape(-1, 1), logsf[i:])
            reg_line = lr.predict(logx[i:].reshape(-1, 1))
            r2 = 1 - np.var(logsf[i:] - reg_line)/np.var(logsf[i:])
            self.r2_adj[i] = 1 - ((1-r2)*(n-i-1)/(n-i-2))

        self.iopt = np.argmax(self.r2_adj)
        self.lr = LinearRegression()
        self.lr.fit(logx[self.iopt:].reshape(-1, 1), logsf[self.iopt:])
        self.c = np.exp(self.lr.intercept_)
        self.alpha = -self.lr.coef_[0]
        return self
        
    def loglog_plot(self, ax=None):
        """Plots the survival function and the fitted power law model in log-log scale.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes to draw the plot on. If not provided, a new figure and axes are created.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 9))
            
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.scatter(self.x, self.sf, alpha=0.5)
        ylim = np.array(ax.get_ylim())
        ax.plot((ylim/self.c)**(-1/self.alpha), ylim, 'r', label="power law α = {0:.2}".format(self.alpha))
        ax.set_ylim(ylim)
        ax.grid(True, which="both", ls="-")
        ax.legend()
