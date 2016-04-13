from .base import (AlphaAngleMetricBase, ContactMetricBase, DihedralMetricBase,
                   MetricBase)
from ..core import mi, nmi

import numpy as np
from itertools import combinations_with_replacement as combinations

from multiprocessing import Pool
from contextlib import closing

__all__ = ['AlphaAngleMutualInformation', 'ContactMutualInformation',
           'DihedralMutualInformation']


class MutualInformationBase(MetricBase):

    """Base mutual information object"""

    def _partial_mutinf(self, p):
        i, j = p

        return self._est(self.n_bins,
                         self.data[i].values.T,
                         self.data[j].values.T,
                         rng=self.rng,
                         method=self.method)

    def _mutinf(self):

        idx = np.triu_indices(self.labels.size)
        M = np.zeros((self.labels.size, self.labels.size))

        with closing(Pool(processes=self.n_threads)) as pool:
            M[idx] = list(pool.map(self._partial_mutinf,
                                   combinations(self.labels, 2)))
            pool.terminate()

        M[idx[::-1]] = M[idx]

        return M

    def partial_transform(self, traj, shuffled=False):
        self.data = self._extract_data(traj)
        self.labels = np.unique(self.data.columns.levels[0])
        if shuffled:
            self._shuffle()

        return self._mutinf()

    def __init__(self, normed=False, **kwargs):
        self.data = None
        self._est = nmi if normed else mi

        super(MutualInformationBase, self).__init__(**kwargs)


class AlphaAngleMutualInformation(AlphaAngleMetricBase, MutualInformationBase):

    """Mutual information calculations for alpha angles"""


class ContactMutualInformation(ContactMetricBase, MutualInformationBase):

    """Mutual information calculations for contacts"""


class DihedralMutualInformation(DihedralMetricBase, MutualInformationBase):

    """Mutual information calculations for dihedral angles"""
