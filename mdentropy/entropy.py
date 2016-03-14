import numpy as np
from scipy.stats import entropy as naive
from scipy.stats.kde import gaussian_kde as gkde
from scipy.special import psi
from mdentropy.utils import hist


<<<<<<< HEAD
def ent(*args, r=None, method='chaowangjost'):
    bins = hist(*args, r=r)
=======
def ent(nbins, r, method, *args):
    if method == 'kde':
        return kde(r, *args)

    bins = hist(nbins, r, *args)

>>>>>>> master
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


<<<<<<< HEAD
def mi(X, Y, r=[-180., 180.], method='chaowangjost'):
    return (ent(X, r=[r], method=method) +
            ent(Y, r=[r], method=method) -
            ent(X, Y, r=2*[r], method=method))


def nmi(X, Y, r=[-180., 180.], method='chaowangjost'):
    return np.nan_to_num(mi(X, Y, r=r, method=method) /
                         np.sqrt(ent(X, r=[r], method=method) *
                         ent(Y, r=[r], method=method)))


def ce(X, Y, r=[-180., 180.], method='chaowangjost'):
    return (ent(X, Y, r=2*[r], method=method) -
            ent(Y, r=[r], method=method))


def cmi(X, Y, Z, r=[-180., 180.], method='chaowangjost'):
    return (ent(X, Z, r=2*[r], method=method) +
            ent(Y, Z, r=2*[r], method=method) -
            ent(Z, r=[r], method=method) -
            ent(X, Y, Z, r=3*[r], method=method))


def ncmi(X, Y, Z, r=[-180., 180.], method='chaowangjost'):
    return np.nan_to_num(1 + (ent(Y, Z, r=2*[r], method=method) -
                              ent(X, Y, Z, r=3*[r], method=method)) /
                         ce(X, Z, r=r, method=method))
=======
def mi(nbins, X, Y, r=[-180., 180.], method='kde'):
    return (ent(nbins, [r], method, X) +
            ent(nbins, [r], method, Y) -
            ent(nbins, 2*[r], method, X, Y))


def nmi(nbins, X, Y, r=[-180., 180.], method='kde'):
    return np.nan_to_num(mi(nbins, X, Y, r=r) /
                         np.sqrt(ent(nbins, [r], method, X) *
                         ent(nbins, [r], method, Y)))


def ce(nbins, X, Y, r=[-180., 180.], method='kde'):
    return (ent(nbins, 2*[r], method, X, Y) -
            ent(nbins, [r], method, Y))


def cmi(nbins, X, Y, Z, r=[-180., 180.], method='kde'):
    return (ent(nbins, 2*[r], method, X, Z) +
            ent(nbins, 2*[r], method, Y, Z) -
            ent(nbins, [r], method, Z) -
            ent(nbins, 3*[r], method, X, Y, Z))


def ncmi(nbins, X, Y, Z, r=[-180., 180.], method='kde'):
    return np.nan_to_num(1 + (ent(nbins, 2*[r], method, Y, Z) -
                         ent(nbins, 3*[r], method, X, Y, Z)) /
                         ce(nbins, X, Z, r=r, method=method))
>>>>>>> master
