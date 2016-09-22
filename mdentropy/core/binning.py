from ..utils import unique_row_count

from numpy import (array, atleast_1d, digitize, empty, floor, linspace, log2,
                   histogramdd, hstack, ndarray, sqrt, vstack)
from scipy.stats import skew

__all__ = ['hist', 'symbolic', 'doanes_rule']


def doanes_rule(x):
    """Convenience function for choosing an optimal number of bins using Doane's Rule.

    Parameters
    ----------
    x : numpy.ndarray or list of floats
        Data to be binned.

    Returns
    -------
    n_bins : int
    """
    if not isinstance(x, ndarray):
        x = array(x)

    n = x.shape[0]
    g1 = atleast_1d(skew(x))
    sg1 = sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))

    return min(floor(1 + log2(n) + log2(1 + abs(g1)/sg1)))


def hist(n_bins, rng, *args):
    """Convenience function for histogramming N-dimentional data

    Parameters
    ----------
    n_bins : int
        Number of bins.
    rng : list of lists
        List of min/max values to bin data over.
    args : array_like, shape = (n_samples, )
        Data of which to histogram.

    Returns
    -------
    bins : array_like, shape = (n_bins, )
    """
    data = vstack((args)).T

    if n_bins is None:
        n_bins = doanes_rule(data)

    return histogramdd(data, bins=n_bins, range=rng)[0].flatten()


def symbolic(n_bins, rng, *args):
    """Symbolic binning of data

    Parameters
    ----------
    rng : list of lists
        List of min/max values for each dimention.
    n_bins : int
        Number of bins to use.
    args : array_like, shape = (n_samples, )
        Data of which to calculate entropy. Each array must have the same
        number of samples.

    Returns
    -------
    counts : float
    """
    labels = empty(0).reshape(args[0].shape[0], 0)
    if n_bins is None:
        n_bins = min(map(doanes_rule, args))

    for i, arg in enumerate(args):

        partitions = linspace(rng[i][0], rng[i][1], n_bins + 1)
        label = digitize(arg, partitions).reshape(-1, 1)

        labels = hstack((labels, label))

    return unique_row_count(labels)
