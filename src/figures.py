import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    
    
def plot_pdf_hist(distr, ax=None, **params):
    """
    Plots a histogram of random samples and the PDF of a given distribution.
    
    Args:
        distr (scipy.stats.rv_continuous): A continuous distribution from scipy.stats.
        ax (matplotlib.axes.Axes, optional): A matplotlib axes object to plot on. If not provided, a new figure will be created.
        **params: Additional keyword arguments for the distribution's methods (e.g., loc, scale, etc.).
        
    Returns:
        None
    """
    random_samples = distr.rvs(size=10000, **params)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
        
    ax.hist(random_samples, bins=100, density=True, alpha=0.5, label='Random samples')
    xlim = ax.get_xlim()
    x = np.linspace(xlim[0], xlim[1], 1000)
    pdf_values = distr.pdf(x, **params)
    ax.plot(x, pdf_values, label='PDF')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.legend()


def plot_cdf_hist(distr, ax=None, **params):
    """
    Plots the empirical CDF from random samples and the actual CDF of a given distribution.
    
    Args:
        distr (scipy.stats.rv_continuous): A continuous distribution from scipy.stats.
        ax (matplotlib.axes.Axes, optional): A matplotlib axes object to plot on. If not provided, a new figure will be created.
        **params: Additional keyword arguments for the distribution's methods (e.g., loc, scale, etc.).
        
    Returns:
        None
    """
    random_samples = distr.rvs(size=5000, **params)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
        
    # Compute the empirical CDF
    sorted_samples = np.sort(random_samples)
    empirical_cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)

    ax.plot(sorted_samples, empirical_cdf, label='Empirical CDF', marker='o', alpha=0.5, linestyle='none')
    
    xlim = ax.get_xlim()
    x = np.linspace(xlim[0], xlim[1], 1000)
    cdf_values = distr.cdf(x, **params)
    ax.plot(x, cdf_values, label='CDF', linestyle='-', lw=2)
    
    ax.set_xlabel('x')
    ax.set_ylabel('Cumulative Probability')
    ax.legend()
