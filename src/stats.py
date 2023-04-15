import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import powerlaw, norm, t, cauchy, pareto
from sklearn.linear_model import LinearRegression, HuberRegressor

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


def compare_histograms(x, y, ax, bins=30, alpha=0.5):
    """
    Compare the histograms of two numpy arrays x and y with transparent bars.

    Args:
        x (array_like): First input array.
        y (array_like): Second input array.
        ax (matplotlib.axes.Axes): Matplotlib Axes object to plot the histograms.
        bins (int): Number of bins for the histogram. Default is 30.
        alpha (float): Transparency level for the histogram bars. Default is 0.5.

    Returns:
        None: Plots the histograms of x and y on the provided Axes object.
    """
    # Compute the range for both arrays
    data_min = min(x.min(), y.min())
    data_max = max(x.max(), y.max())

    # Plot histograms
    ax.hist(x, bins=bins, alpha=alpha, label='x', color='blue', range=(data_min, data_max))
    ax.hist(y, bins=bins, alpha=alpha, label='y', color='red', range=(data_min, data_max))

    ax.legend()

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