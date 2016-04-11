from .binning import symbolic

from itertools import chain

from numpy import ndarray
from numpy import sum as npsum
from numpy import (arange, bincount, diff, linspace, log, log2,
                   meshgrid, nan_to_num, nansum, product, ravel, reshape,
                   split, vstack)

from scipy.stats import entropy as naive
from scipy.stats.kde import gaussian_kde as kernel
from scipy.special import psi

__all__ = ['ent', 'ce']


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
            rng[i] = (min(arg), max(arg))

    if method == 'kde':
        return kde_ent(rng, *args, gride_size=n_bins or 20)

    counts = symbolic(n_bins, rng, *args)

    if method == 'chaowangjost':
        return chaowangjost(counts)
    elif method == 'grassberger':
        return grassberger(counts)

    return naive(counts)


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
    return (ent(n_bins, 2 * [rng], method, x, y) -
            ent(n_bins, [rng], method, y))


def kde_ent(rng, *args, gride_size=20):
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
    space = [linspace(i[0], i[1], gride_size) for i in rng]
    grid = meshgrid(*tuple(space))
    pr = reshape(gkde(vstack(map(ravel, grid))),
                 n_dims * [gride_size])
    return -nansum(pr * log2(pr)) * product(diff(space)[:, 0])


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
