from ..utils import hist, adaptive, unique_row_count

from itertools import chain

from numpy import ndarray
from numpy import sum as npsum
from numpy import (arange, bincount, diff, digitize, empty, hstack, linspace,
                   log, log2, meshgrid, nan_to_num, nansum, product, ravel,
                   reshape, split, vstack)

from scipy.stats import entropy as naive
from scipy.stats.kde import gaussian_kde as kernel
from scipy.special import psi


def ent(n_bins, rng, method, *args):
    """Entropy calculation

    Parameters
    ----------
    n_bins : int
        Number of bins.
    rng : list of lists
        List of min/max values to bin data over.
    method : {'kde', 'chaowangjost', 'grassberger', None}
        Method used to calculate entropy.
    args : numpy.ndarray, shape = (n_samples, ) or (n_features, n_samples)
        Data of which to calculate entropy. Each array must have the same
        number of samples.
    Returns
    -------
    entropy : float
    """

    args = list(chain(*[map(ndarray.flatten, split(arg, arg.shape[0]))
                        if arg.ndim == 2
                        else [arg]
                        for arg in args]))

    if rng is None or None in rng:
        rng = len(args)*[None]
    for i, arg in enumerate(args):
        if rng[i] is None:
            rng[i] = [min(arg), max(arg)]

    if method == 'kde':
        return kde(rng, *args, gride_size=n_bins or 20)

    if method == 'symbolic':
        return symbolic(n_bins, rng, *args)

    if n_bins and method != 'adaptive':
        bins = hist(n_bins, rng, *args)
    else:
        bins = adaptive(rng=rng, *args)

    if method == 'chaowangjost':
        return chaowangjost(bins)
    elif method in ['grassberger', 'adaptive']:
        return grassberger(bins)
    return naive(bins)


def symbolic(n_bins, rng, *args):
    """Entropy calculation using symbolic entropy estimation.

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
    entropy : float
    """

    labels = empty(0).reshape(args[0].shape[0], 0)
    for i, arg in enumerate(args):
        partitions = linspace(rng[i][0], rng[i][1], n_bins+1)
        label = digitize(arg, partitions).reshape(-1, 1)
        labels = hstack((labels, label))

    bins = unique_row_count(labels)

    return grassberger(bins)


def kde(rng, *args, gride_size=20):
    """Entropy calculation using Gaussian kernel density estimation.

    Parameters
    ----------
    rng : list of lists
        List of min/max values for each dimention.
    args : array_like, shape = (n_samples, )
        Data of which to calculate entropy. Each array must have the same
        number of samples.
    grid_size : int
        Number of partitions along a dimension in the meshgrid.
    Returns
    -------
    entropy : float
    """
    n_dims = len(args)
    data = vstack((args))
    gkde = kernel(data)
    x = [linspace(i[0], i[1], gride_size) for i in rng]
    grid = meshgrid(*tuple(x))
    z = reshape(gkde(vstack(map(ravel, grid))),
                n_dims*[gride_size])
    return -nansum(z*log2(z))*product(diff(x)[:, 0])


def grassberger(bins):
    """Entropy calculation using Grassberger correction.
    doi:10.1016/0375-9601(88)90193-4

    Parameters
    ----------
    bins : list
        Binned data
    Returns
    -------
    entropy : float
    """
    n = npsum(bins)
    return npsum(bins*(log(n) -
                       nan_to_num(psi(bins)) -
                       ((-1.)**bins/(bins + 1.))))/n


def chaowangjost(bins):
    """Entropy calculation using Chao, Wang, Jost correction.
    doi: 10.1111/2041-210X.12108

    Parameters
    ----------
    bins : list
        Binned data
    Returns
    -------
    entropy : float
    """
    n = npsum(bins)
    bc = bincount(bins.astype(int))
    if bc[2] == 0:
        if bc[1] == 0:
            A = 1.
        else:
            A = 2./((n - 1.) * (bc[1] - 1.) + 2.)
    else:
        A = 2. * bc[2]/((n - 1.) * (bc[1] - 1.) + 2. * bc[2])
    p = arange(1, int(n))
    p = 1./p * (1. - A)**p
    cwj = npsum(bins/n * (psi(n) - nan_to_num(psi(bins))))
    if bc[1] > 0 and A != 1.:
        cwj += nan_to_num(bc[1]/n *
                          (1 - A)**(1 - n * (-log(A) - npsum(p))))
    return cwj


def ce(n_bins, x, y, rng=None, method='kde'):
    """Condtional entropy calculation

    Parameters
    ----------
    n_bins : int
        Number of bins.
    x : array_like, shape = (n_samples, )
        Conditioned variable.
    y : array_like, shape = (n_samples, )
        Conditional variable.
    rng : list
        List of min/max values to bin data over.
    method : {'kde', 'chaowangjost', 'grassberger', None}
        Method used to calculate entropy.
    Returns
    -------
    entropy : float
    """
    return (ent(n_bins, 2*[rng], method, x, y) -
            ent(n_bins, [rng], method, y))
