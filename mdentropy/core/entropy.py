from .binning import symbolic
from ..utils import kde, nearest_distances

from itertools import chain

from numpy import ndarray
from numpy import sum as npsum
from numpy import (atleast_2d, arange, bincount, diff, finfo, float32,
                   hsplit, log, nan_to_num, nansum, product, ravel, vstack)

from scipy.stats import entropy as naive
from scipy.special import psi

__all__ = ['entropy', 'centropy']
EPS = finfo(float32).eps


def entropy(n_bins, rng, method, *args):
    """Entropy calculation

    Parameters
    ----------
    n_bins : int
        Number of bins.
    rng : list of lists
        List of min/max values to bin data over.
    method : {'kde', 'chaowangjost', 'grassberger', 'knn', None}
        Method used to calculate entropy.
    args : numpy.ndarray, shape = (n_samples, ) or (n_samples, n_dims)
        Data of which to calculate entropy. Each array must have the same
        number of samples.

    Returns
    -------
    entropy : float
    """
    args = [args] if isinstance(args, ndarray) else args
    args = list(chain(*[map(ravel, hsplit(arg, arg.shape[1]))
                        if arg.ndim == 2
                        else atleast_2d(arg)
                        for arg in args]))

    if method == 'knn':
        return knn_entropy(*args, k=n_bins)

    if rng is None or None in rng:
        rng = len(args) * [None]

    for i, arg in enumerate(args):
        if rng[i] is None:
            rng[i] = (min(arg), max(arg))

    if method == 'kde':
        return kde_entropy(rng, *args, grid_size=n_bins or 20)

    counts = symbolic(n_bins, rng, *args)

    if method == 'chaowangjost':
        return chaowangjost(counts)
    elif method == 'grassberger':
        return grassberger(counts)

    return naive(counts)


def centropy(n_bins, x, y, rng=None, method='knn'):
    """Conditional entropy calculation

    Parameters
    ----------
    n_bins : int
        Number of bins.
    x : array_like, shape = (n_samples, n_dims)
        Conditioned variable.
    y : array_like, shape = (n_samples, n_dims)
        Conditional variable.
    rng : list
        List of min/max values to bin data over.
    method : {'kde', 'chaowangjost', 'grassberger', None}
        Method used to calculate entropy.

    Returns
    -------
    entropy : float
    """
    return (entropy(n_bins, 2 * [rng], method, x, y) -
            entropy(n_bins, [rng], method, y))


def knn_entropy(*args, k=None):
    """Entropy calculation

    Parameters
    ----------
    args : numpy.ndarray, shape = (n_samples, ) or (n_samples, n_dims)
        Data of which to calculate entropy. Each array must have the same
        number of samples.
    k : int
        Number of bins.

    Returns
    -------
    entropy : float
    """
    data = vstack((args)).T
    n_samples, n_dims = data.shape
    k = k if k else max(3, int(n_samples * 0.01))

    nneighbor = nearest_distances(data, k=k)
    const = psi(n_samples) - psi(k) + n_dims * log(2.)

    return (const + n_dims * log(nneighbor).mean())


def kde_entropy(rng, *args, grid_size=20, **kwargs):
    """Kernel Density Estimation of Entropy"""
    data = vstack((args)).T

    prob, space = kde(data, rng, grid_size=20, **kwargs)

    return -nansum(prob * log(prob)) * product(diff(space)[:, 0])


def grassberger(counts):
    """Entropy calculation using Grassberger correction.
    doi:10.1016/0375-9601(88)90193-4

    Parameters
    ----------
    counts : list
        bin counts

    Returns
    -------
    entropy : float
    """
    n_samples = npsum(counts)
    return npsum(counts * (log(n_samples) -
                           nan_to_num(psi(counts)) -
                           ((-1.) ** counts / (counts + 1.)))) / n_samples


def chaowangjost(counts):
    """Entropy calculation using Chao, Wang, Jost correction.
    doi: 10.1111/2041-210X.12108

    Parameters
    ----------
    counts : list
        bin counts

    Returns
    -------
    entropy : float
    """
    n_samples = npsum(counts)
    bcbc = bincount(counts.astype(int))
    if len(bcbc) < 3:
        return grassberger(counts)
    if bcbc[2] == 0:
        if bcbc[1] == 0:
            A = 1.
        else:
            A = 2. / ((n_samples - 1.) * (bcbc[1] - 1.) + 2.)
    else:
        A = 2. * bcbc[2] / ((n_samples - 1.) * (bcbc[1] - 1.) +
                            2. * bcbc[2])
    pr = arange(1, int(n_samples))
    pr = 1. / pr * (1. - A) ** pr
    entropy = npsum(counts / n_samples * (psi(n_samples) -
                    nan_to_num(psi(counts))))

    if bcbc[1] > 0 and A != 1.:
        entropy += nan_to_num(bcbc[1] / n_samples *
                              (1 - A) ** (1 - n_samples *
                                          (-log(A) - npsum(pr))))
    return entropy
