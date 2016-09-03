from ..utils import unique_row_count

from itertools import product as iterproduct

from numpy import (digitize, empty, linspace, histogramdd, hstack, product,
                   vstack, zeros)

from scipy.stats import binom_test

__all__ = ['hist', 'symbolic', 'adaptive']


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
    for i, arg in enumerate(args):
        if n_bins is not None:
            partitions = linspace(rng[i][0], rng[i][1], n_bins + 1)
            label = digitize(arg, partitions).reshape(-1, 1)
        else:
            rng = tuple(rng)
            label = adaptive(arg)
        labels = hstack((labels, label))

    return unique_row_count(labels)


def adaptive(*args, rng=None, alpha=0.05):
    """Darbellay-Vajda adaptive partitioning (doi:10.1109/18.761290)

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
    data = vstack(args).T

    # Get number of dimensions
    n_dims = data.shape[1]
    dims = range(n_dims)

    # If no ranges are supplied, initialize with min/max for each dimension
    if rng is None:
        rng = tuple((data[:, i].min(), data[:, i].max()) for i in dims)

    if not (0. <= alpha < 1):
        raise ValueError('alpha must be a float in [0, 1).')

    def dvpartition(data, rng):
        nonlocal n_dims
        nonlocal counts
        nonlocal labels
        nonlocal dims

        # Filter out data that is not in our initial partition
        where = product([(i[0] <= data[:, j]) * (i[1] >= data[:, j])
                        for j, i in enumerate(rng)], 0).astype(bool)
        filtered = data[where, :]

        # Subdivide our partitions by the midpoint in each dimension
        partitions = set([])
        part = [linspace(rng[i][0], rng[i][1], 3) for i in dims]
        newrng = set((tuple((part[i][j[i]], part[i][j[i] + 1]) for i in dims)
                     for j in iterproduct(*(n_dims * [[0, 1]]))),)

        # Calculate counts for new partitions
        freq = histogramdd(filtered, bins=part)[0]

        # Perform binomial test which a given alpha,
        # and if not uniform proceed
        if (binom_test(freq) < alpha / 2. and
                False not in ((filtered.max(0) - filtered.min(0)).T > 0)):

            # For each new partition continue algorithm recursively
            for nr in newrng:
                newpart = dvpartition(data, rng=nr)
                for newp in newpart:
                    partitions.update(tuple((newp,)))

        # Else if uniform and contains data, return current partition
        elif filtered.shape[0] > 0:
            partitions = set(tuple((rng,)))
            labels[where] = len(counts)
            counts += (filtered.shape[0],)
        return partitions

    counts = ()
    labels = zeros(data.shape[0], dtype=int)
    dvpartition(data, rng)
    return labels.reshape(-1, n_dims)
