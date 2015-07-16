import numpy as np
from scipy import stats
from scipy.special import psi
from sklearn.metrics import mutual_info_score
from mdentropy.utils import hist


def ent(nbins, r, *args):
    bins = hist(nbins, r, *args)
    return stats.entropy(bins)


def entc(nbins, r, *args):
    N = args[0].shape[0]
    bins = hist(nbins, r, *args)
    return np.sum(bins*(np.log(N)
                  - np.nan_to_num(psi(bins))
                  - ((-1)**bins/(bins + 1))))/N


def mi(nbins, X, Y, r=[-180., 180.]):
    return (entc(nbins, [r], X)
           + entc(nbins, [r], Y)
           - entc(nbins, 2*[r], X, Y))

def nmi(nbins, X, Y, r=[-180., 180.]):
    return (mi(nbins, X, Y, r = 2*[r])/
            np.sqrt(entc(nbins, [r], X)*entc(nbins, [r], Y)))


def ce(nbins, X, Y, r=[-180., 180.]):
    return (entc(nbins, 2*[r], X, Y)
           - entc(nbins, [r], Y))


def cmi(nbins, X, Y, Z, r=[-180., 180.]):
    return (entc(nbins, 2*[r], X, Z)
            +entc(nbins, 2*[r], Y, Z)
            - entc(nbins, [r], Z)
            - entc(nbins, 3*[r], X, Y, Z))


def ncmi(nbins, X, Y, Z, r=[-180., 180.]):
    return (1 + (entc(nbins, 2*[r], Y, Z)
            - entc(nbins, 3*[r], X, Y, Z))/ce(nbins, X, Z, r=r))
