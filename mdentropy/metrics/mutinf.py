from .base import MetricBase, DihedralMetricBase
from ..core import mi, nmi

import numpy as np
from itertools import combinations_with_replacement as combinations

from multiprocessing import Pool
from contextlib import closing

__all__ = ['DihedralMutualInformation']


class MutualInformationBase(MetricBase):
    """
    Base mutual information object
    """

    def _partial_mutinf(cls, p):
        i, j = p

        return cls._est(cls.n_bins,
                        cls.data[i].values.T,
                        cls.data[j].values.T,
                        rng=cls.rng,
                        method=cls.method)

    def _mutinf(cls):

        idx = np.triu_indices(cls.labels.size)
        M = np.zeros((cls.labels.size, cls.labels.size))

        with closing(Pool(processes=cls.n_threads)) as pool:
            M[idx] = list(pool.map(cls._partial_mutinf,
                                   combinations(cls.labels, 2)))
            pool.terminate()

        M[idx[::-1]] = M[idx]

        return M

    def partial_transform(cls, traj, shuffled=False):
        cls.data = cls._extract_data(traj)
        cls.labels = np.unique(cls.data.columns.levels[0])
        if shuffled:
            cls._shuffle()

        return cls._mutinf()

    def __init__(cls, normed=False, **kwargs):
        cls.data = None
        cls._est = nmi if normed else mi

        super(MutualInformationBase, cls).__init__(**kwargs)


class DihedralMutualInformation(DihedralMetricBase, MutualInformationBase):
    """
    Mutual information calculations for dihedral angles
    """
