from .entropy import ent, ce
import numpy as np


def mi(nbins, X, Y, r=[-180., 180.], method='kde'):
    return (ent(nbins, [r], method, X) +
            ent(nbins, [r], method, Y) -
            ent(nbins, 2*[r], method, X, Y))


def nmi(nbins, X, Y, r=[-180., 180.], method='kde'):
    return np.nan_to_num(mi(nbins, X, Y, method=method, r=r) /
                         np.sqrt(ent(nbins, [r], method, X) *
                         ent(nbins, [r], method, Y)))


def cmi(nbins, X, Y, Z, r=[-180., 180.], method='kde'):
    return (ent(nbins, 2*[r], method, X, Z) +
            ent(nbins, 2*[r], method, Y, Z) -
            ent(nbins, [r], method, Z) -
            ent(nbins, 3*[r], method, X, Y, Z))


def ncmi(nbins, X, Y, Z, r=[-180., 180.], method='kde'):
    return np.nan_to_num(1 + (ent(nbins, 2*[r], method, Y, Z) -
                         ent(nbins, 3*[r], method, X, Y, Z)) /
                         ce(nbins, X, Z, r=r, method=method))
