from ..core import cmi, ncmi
from .base import (AlphaAngleBaseMetric, ContactBaseMetric, DihedralBaseMetric,
                   BaseMetric)


import numpy as np
from itertools import product

from multiprocessing import Pool
from contextlib import closing

__all__ = ['AlphaAngleTransferEntropy', 'ContactTransferEntropy',
           'DihedralTransferEntropy']


class TransferEntropyBase(BaseMetric):

    """Base transfer entropy object"""

    def _partial_tent(self, p):
        i, j = p

        return self._est(self.n_bins,
                         self.data2[j].values.T,
                         self.shuffled_data[i].values.T,
                         self.shuffled_data[j].values.T,
                         rng=self.rng,
                         method=self.method)

    def _exec(self):
        with closing(Pool(processes=self.n_threads)) as pool:
            CMI = list(pool.map(self._partial_tent,
                                product(self.labels, self.labels)))
            pool.terminate()

        return np.reshape(CMI, (self.labels.size, self.labels.size)).T

    def _before_exec(self, traj):
        traj1, traj2 = traj
        self.data1 = self._extract_data(traj1)
        self.data2 = self._extract_data(traj2)
        self.shuffled_data = self.data1
        self.labels = np.unique(self.data1.columns.levels[0])

    def transform(self, trajs):
        for traj in trajs:
            yield self.partial_transform(traj)

    def __init__(self, normed=True, **kwargs):
        self.data1 = None
        self.data2 = None
        self._est = ncmi if normed else cmi

        super(TransferEntropyBase, self).__init__(**kwargs)


class AlphaAngleTransferEntropy(AlphaAngleBaseMetric, TransferEntropyBase):

    """Mutual information calculations for alpha angles"""


class ContactTransferEntropy(ContactBaseMetric, TransferEntropyBase):

    """Transfer entropy calculations for contacts"""


class DihedralTransferEntropy(DihedralBaseMetric, TransferEntropyBase):

    """Transfer entropy calculations for dihedral angles"""
