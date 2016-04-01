import numpy as np
from scipy.stats import entropy as naive
from scipy.stats.kde import gaussian_kde as gkde
from scipy.special import psi
from mdentropy.utils import hist


def ent(nbins, r, method, *args):
    if method == 'kde':
        return kde(r, *args, gride_size=nbins)

    bins = hist(nbins, r, *args)

    if method == 'chaowangjost':
        return chaowangjost(bins)
    elif method == 'grassberger':
        return grassberger(bins)
    return naive(bins)


def kde(r, *args, gride_size=20):
    N = len(args)
    data = np.vstack((args))
    kde = gkde(data)
    x = [np.linspace(i[0], i[1], gride_size) for i in r]
    G = np.meshgrid(*tuple(x))
    Z = np.reshape(kde(np.vstack(map(np.ravel, G))),
                   N*[gride_size])
    return -np.nansum(Z*np.log2(Z))*np.product(np.diff(x)[:, 0])


def grassberger(bins):
    N = np.sum(bins)
    return np.sum(bins*(np.log(N) -
                        np.nan_to_num(psi(bins)) -
                        ((-1.)**bins/(bins + 1.))))/N


def chaowangjost(bins):
    N = np.sum(bins)
    bc = np.bincount(bins.astype(int))
    if bc[2] == 0:
        if bc[1] == 0:
            A = 1.
        else:
            A = 2./((N - 1.) * (bc[1] - 1.) + 2.)
    else:
        A = 2. * bc[2]/((N - 1.) * (bc[1] - 1.) + 2. * bc[2])
    p = np.arange(1, int(N))
    p = 1./p * (1. - A)**p
    cwj = np.sum(bins/N * (psi(N) - np.nan_to_num(psi(bins))))
    if bc[1] > 0 and A != 1.:
        cwj += np.nan_to_num(bc[1]/N *
                             (1 - A)**(1 - N * (-np.log(A) - np.sum(p))))
    return cwj


def ce(nbins, X, Y, r=[-180., 180.], method='kde'):
    return (ent(nbins, 2*[r], method, X, Y) -
            ent(nbins, [r], method, Y))
