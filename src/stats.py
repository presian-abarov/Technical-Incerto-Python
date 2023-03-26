import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import powerlaw, norm, t, cauchy, pareto
from sklearn.linear_model import LinearRegression, HuberRegressor

def median_absolute_deviation(x):
    """
    Compute the median absolute deviation of an array x.

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    mad : float
        Median absolute deviation of x, adjusted to be 1 for standard normal distribution.

    """
    # Compute the median of x
    med = np.median(x)
    
    # Compute the median absolute deviation of x, and adjust for standard normal distribution
    mad = np.median(np.abs(x - med)) / 0.6745
    
    return mad


def detect_simple_regression_outliers(x, y, threshold=2.0):
    """
    Detect outliers of a simple linear regression model using robust regression and z-scores.

    Parameters
    ----------
    x : array_like
        Predictor variable.
    y : array_like
        Response variable.
    threshold : float, optional
        Threshold for z-scores. Observations with z-scores greater than this threshold are considered outliers.
        Default is 2.0.

    Returns
    -------
    outliers : ndarray
        Indices of outlier observations.

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