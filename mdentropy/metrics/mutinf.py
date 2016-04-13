from ..core import mi, nmi
from .base import (AlphaAngleBaseMetric, ContactBaseMetric, DihedralBaseMetric,
                   BaseMetric)

import numpy as np
from itertools import combinations_with_replacement as combinations

from multiprocessing import Pool
from contextlib import closing

__all__ = ['AlphaAngleMutualInformation', 'ContactMutualInformation',
           'DihedralMutualInformation']


class MutualInformationBase(BaseMetric):

    """Base mutual information object"""

    def _partial_mutinf(self, p):
        i, j = p

        return self._est(self.n_bins,
                         self.data[i].values.T,
                         self.shuffled_data[j].values.T,
                         rng=self.rng,
                         method=self.method)

    def _exec(self):
        uidx = np.triu_indices(self.labels.size)
        lidx = np.tril_indices(self.labels.size)
        M = np.zeros((self.labels.size, self.labels.size))

        with closing(Pool(processes=self.n_threads)) as pool:
            M[uidx] = list(pool.map(self._partial_mutinf,
                                    combinations(self.labels, 2)))
            pool.terminate()

        M[lidx] = M[uidx]

        return M

    def __init__(self, normed=True, **kwargs):
        self.data = None
        self._est = nmi if normed else mi

        super(MutualInformationBase, self).__init__(**kwargs)


class AlphaAngleMutualInformation(AlphaAngleBaseMetric, MutualInformationBase):

    """Mutual information calculations for alpha angles"""


class ContactMutualInformation(ContactBaseMetric, MutualInformationBase):

    """Mutual information calculations for contacts"""


class DihedralMutualInformation(DihedralBaseMetric, MutualInformationBase):

    """Mutual information calculations for dihedral angles"""
