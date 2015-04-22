import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score


def ent(nbins, range, correction=True, *args):
    data = np.vstack((args)).T
    hist = np.histogramdd(data, bins=nbins, range=range)
    dx = hist[1][0][1] - hist[1][0][0]
    H = stats.entropy(hist[0].flatten())
    if correction:
        return H + data.shape[1]*np.log(dx)
    return H


def mi(nbins, X, Y, range=2*[[-180., 180.]]):
    H = np.histogram2d(X, Y, bins=nbins, range=range)
    return mutual_info_score(None, None, contingency=H[0])


def ce(nbins, X, Y, range=[-180., 180.]):
    return ent(nbins, 2*[range], X, Y) - ent(nbins, [range], Y)


def cmi(nbins, X, Y, Z, range=[-180., 180.]):
    return sum([ent(nbins, 2*[range], X, Z),
                ent(nbins, 2*[range], Y, Z),
                -ent(nbins, 3*[range], X, Y, Z),
                - ent(nbins, [range], Z)])
