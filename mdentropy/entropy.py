import numpy as np
from scipy import stats
from scipy.special import psi
from sklearn.metrics import mutual_info_score
from mdentropy.utils import hist


def ent(nbins, range, *args):
    bins = hist(nbins, range, *args)
    return stats.entropy(bins)


def entc(nbins, range, *args):
    N = args[0].shape[0]
    bins = hist(nbins, range, *args)
    return np.sum(bins*(np.log(N)
                  - psi(bins)
                  - ((-1)**bins/(bins + 1))))/N


def mi(nbins, X, Y, range=2*[[-180., 180.]]):
    bins = hist(nbins, range, X, Y)
    return entc(nbins, range, X)
           + entc(nbins, range, Y)
           - entc(nbins, range, X, Y)


def ce(nbins, X, Y, range=[-180., 180.]):
    return entc(nbins, 2*[range], X, Y)
           - entc(nbins, [range], Y)


def cmi(nbins, X, Y, Z, range=[-180., 180.]):
    return sum([entc(nbins, 2*[range], X, Z),
                entc(nbins, 2*[range], Y, Z),
                - entc(nbins, [range], Z),
                - entc(nbins, 3*[range], X, Y, Z)])


def ncmi(nbins, X, Y, Z, range=[-180., 180.]):
    return (1 + (entc(nbins, 2*[range], Y, Z)
            - entc(nbins, 3*[range], X, Y, Z))/ce(nbins, X, Z, range=range))
