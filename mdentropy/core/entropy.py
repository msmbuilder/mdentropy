from ..utils import hist, adaptive

import numpy as np
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
    args : array_like, shape = (n_samples, )
        Data of which to calculate entropy. Each array must have the same
        number of samples.
    Returns
    -------
    entropy : float
    """
    for i, arg in enumerate(args):
        if rng[i] is None:
            rng[i] = [min(arg), max(arg)]

    if method == 'kde':
        return kde(rng, *args, gride_size=n_bins or 20)

    if n_bins:
        bins = hist(n_bins, rng, *args)
    else:
        bins = adaptive(rng=rng, *args)

    if method == 'chaowangjost':
        return chaowangjost(bins)
    elif method == 'grassberger':
        return grassberger(bins)
    return naive(bins)


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
    data = np.vstack((args))
    gkde = kernel(data)
    x = [np.linspace(i[0], i[1], gride_size) for i in rng]
    grid = np.meshgrid(*tuple(x))
    z = np.reshape(gkde(np.vstack(map(np.ravel, grid))),
                   n_dims*[gride_size])
    return -np.nansum(z*np.log2(z))*np.product(np.diff(x)[:, 0])


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
    n = np.sum(bins)
    return np.sum(bins*(np.log(n) -
                        np.nan_to_num(psi(bins)) -
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
    n = np.sum(bins)
    bc = np.bincount(bins.astype(int))
    if bc[2] == 0:
        if bc[1] == 0:
            A = 1.
        else:
            A = 2./((n - 1.) * (bc[1] - 1.) + 2.)
    else:
        A = 2. * bc[2]/((n - 1.) * (bc[1] - 1.) + 2. * bc[2])
    p = np.arange(1, int(n))
    p = 1./p * (1. - A)**p
    cwj = np.sum(bins/n * (psi(n) - np.nan_to_num(psi(bins))))
    if bc[1] > 0 and A != 1.:
        cwj += np.nan_to_num(bc[1]/n *
                             (1 - A)**(1 - n * (-np.log(A) - np.sum(p))))
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
