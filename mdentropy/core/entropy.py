import numpy as np
from scipy.stats import entropy as naive
from scipy.stats.kde import gaussian_kde as kernel
from scipy.special import psi
from ..utils import hist


def ent(nbins, rng, method, *args):
    for i, arg in enumerate(args):
        if rng[i] is None:
            rng[i] = [min(arg), max(arg)]

    if method == 'kde':
        return kde(rng, *args, gride_size=nbins)

    bins = hist(nbins, rng, *args)

    if method == 'chaowangjost':
        return chaowangjost(bins)
    elif method == 'grassberger':
        return grassberger(bins)
    return naive(bins)


def kde(rng, *args, gride_size=20):
    n_dims = len(args)
    data = np.vstack((args))
    gkde = kernel(data)
    x = [np.linspace(i[0], i[1], gride_size) for i in rng]
    grid = np.meshgrid(*tuple(x))
    z = np.reshape(gkde(np.vstack(map(np.ravel, grid))),
                   n_dims*[gride_size])
    return -np.nansum(z*np.log2(z))*np.product(np.diff(x)[:, 0])


def grassberger(bins):
    n = np.sum(bins)
    return np.sum(bins*(np.log(n) -
                        np.nan_to_num(psi(bins)) -
                        ((-1.)**bins/(bins + 1.))))/n


def chaowangjost(bins):
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


def ce(nbins, x, y, rng=None, method='kde'):
    return (ent(nbins, 2*[rng], method, x, y) -
            ent(nbins, [rng], method, y))
