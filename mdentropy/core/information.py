from .entropy import ent, ce
import numpy as np


def mi(nbins, X, Y, range=None, method='kde'):
    return (ent(nbins, [range], method, X) +
            ent(nbins, [range], method, Y) -
            ent(nbins, 2*[range], method, X, Y))


def nmi(nbins, X, Y, range=None, method='kde'):
    return np.nan_to_num(mi(nbins, X, Y, method=method, range=range) /
                         np.sqrt(ent(nbins, [range], method, X) *
                         ent(nbins, [range], method, Y)))


def cmi(nbins, X, Y, Z, range=None, method='kde'):
    return (ent(nbins, 2*[range], method, X, Z) +
            ent(nbins, 2*[range], method, Y, Z) -
            ent(nbins, [range], method, Z) -
            ent(nbins, 3*[range], method, X, Y, Z))


def ncmi(nbins, X, Y, Z, range=None, method='kde'):
    return np.nan_to_num(1 + (ent(nbins, 2*[range], method, Y, Z) -
                         ent(nbins, 3*[range], method, X, Y, Z)) /
                         ce(nbins, X, Z, range=range, method=method))
