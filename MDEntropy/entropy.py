import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score

def ent(nbins, range=[-180, 180], *args):
    W = np.vstack((args)).T
    n = len(args)
    H = np.histogramdd(W, bins=nbins, range=n*[range])
    dx = H[1][0][1] - H[1][0][0]
    return stats.entropy(H[0].flatten()) + n*np.log(dx)


def mi(nbins, X, Y, range=[-180, 180]):
    H = np.histogram2d(X, Y, bins=nbins, range=2*[range])
    return mutual_info_score(None, None, contingency=H[0])


def ce(nbins, X, Y):
    return ent(nbins, X, Y) - ent(nbins, Y)


def cmi(nbins, X, Y, Z):
    return sum([ent(nbins, X, Z),
                ent(nbins, Y, Z),
                -ent(nbins, X, Y, Z),
                - ent(nbins, Z)])
