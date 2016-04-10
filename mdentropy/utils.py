from __future__ import print_function

import time
from itertools import product as iterproduct

from numpy import sum as npsum
from numpy import array, linspace, histogramdd, product, random, vstack

from scipy.stats import chi2


class timing(object):
    "Context manager for printing performance"
    def __init__(self, iteration):
        self.iteration = iteration

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, ty, val, tb):
        end = time.time()
        print("Round %d : %0.3f seconds" %
              (self.iteration, end-self.start))
        return False


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
    return histogramdd(data, bins=n_bins, range=rng)[0].flatten()


def adaptive(*args, rng=None, alpha=None):
    """ Darbellay-Vajda adaptive partitioning
    doi:10.1109/18.761290
        Parameters
        ----------
        args : array_like, shape = (n_samples, )
            Data of which to histogram.
        rng : list of lists
            List of min/max values to bin data over.
        alpha : float
            Chi-squared test criterion.
        Returns
        -------
        bins : array_like, shape = (n_bins, )
    """
    X = vstack(args).T

    # Get number of dimensions
    n_dims = X.shape[1]
    dims = range(n_dims)

    # If no ranges are supplied, initialize with min/max for each dimension
    if rng is None:
        rng = tuple((X[:, i].min(), X[:, i].max()) for i in dims)

    if alpha is None:
        alpha = 1/X.shape[0]
    elif not (0. <= alpha < 1):
        raise ValueError('alpha must be a float in [0, 1).')

    # Estimate of X2 statistic
    def sX2(freq):
        return npsum((freq - freq.mean())**2)/freq.mean()

    def dvpartition(X, rng):
        nonlocal n_dims
        nonlocal counts
        nonlocal dims
        nonlocal x2

        # Filter out data that is not in our initial partition
        Y = X[product([(i[0] <= X[:, j])*(i[1] >= X[:, j])
                       for j, i in enumerate(rng)], 0).astype(bool), :]

        # Subdivide our partitions by the midpoint in each dimension
        partitions = set([])
        part = [linspace(rng[i][0], rng[i][1], 3) for i in dims]
        newrng = set((tuple((part[i][j[i]], part[i][j[i]+1]) for i in dims)
                     for j in iterproduct(*(n_dims*[[0, 1]]))),)

        # Calculate counts for new partitions
        freq = histogramdd(Y, bins=part)[0]

        # Compare estimate to X2 statistic at given alpha
        chisq = (sX2(freq) >= x2)

        # If not uniform proceed
        if chisq and False not in ((Y.max(0) - Y.min(0)).T > 0):

            # For each new partition continue algorithm recursively
            for nr in newrng:
                newpart = dvpartition(X, rng=nr)
                for newp in newpart:
                    partitions.update(tuple((newp,)))

        # Else if uniform and contains data, return current partition
        elif Y.shape[0] > 0:
            partitions = set(tuple((rng,)))
            counts += (Y.shape[0],)
        return partitions

    counts = ()
    x2 = chi2.ppf(1-alpha, 2**n_dims-1)
    dvpartition(X, rng)
    return array(counts).astype(int)


def shuffle(df, n=1):
    """Convenience function for shuffling values in DataFrame objects

    Parameters
    ----------
    df : pandas.DataFrame
        pandas DataFrame
    n : int
        Number of shuffling iterations.
    Returns
    -------
    sdf : array_like, shape = (n_bins, )
        shuffled DataFrame
    """
    sdf = df.copy()
    sampler = random.permutation
    for _ in range(n):
        sdf = sdf.apply(sampler, axis=0)
        sdf = sdf.apply(sampler, axis=1)
    return sdf
