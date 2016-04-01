from .entropy import ent, ce
import numpy as np


def mi(nbins, x, y, rng=None, method='kde'):
    return (ent(nbins, [rng], method, x) +
            ent(nbins, [rng], method, y) -
            ent(nbins, 2*[rng], method, x, y))


def nmi(nbins, x, y, rng=None, method='kde'):
    return np.nan_to_num(mi(nbins, x, y, method=method, rng=rng) /
                         np.sqrt(ent(nbins, [rng], method, x) *
                         ent(nbins, [rng], method, y)))


def cmi(nbins, x, y, z, rng=None, method='kde'):
    return (ent(nbins, 2*[rng], method, x, z) +
            ent(nbins, 2*[rng], method, y, z) -
            ent(nbins, [rng], method, z) -
            ent(nbins, 3*[rng], method, x, y, z))


def ncmi(nbins, x, y, z, rng=None, method='kde'):
    return np.nan_to_num(1 + (ent(nbins, 2*[rng], method, y, z) -
                         ent(nbins, 3*[rng], method, x, y, z)) /
                         ce(nbins, x, z, rng=rng, method=method))
