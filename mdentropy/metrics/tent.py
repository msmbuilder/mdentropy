from ..core import cmi, ncmi
from .base import (AlphaAngleMetricBase, ContactMetricBase, DihedralMetricBase,
                   MetricBase)


import numpy as np
from itertools import product

from multiprocessing import Pool
from contextlib import closing

__all__ = ['AlphaAngleTransferEntropy', 'ContactTransferEntropy',
           'DihedralTransferEntropy']


class TransferEntropyBase(MetricBase):

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

    def partial_transform(cls, traj, shuffle=0):
        traj1, traj2 = traj
        cls.data1 = cls._extract_data(traj1)
        cls.data2 = cls._extract_data(traj2)
        cls.shuffled_data = cls.data1
        cls.labels = np.unique(cls.data1.columns.levels[0])

        result = cls._exec()
        for _ in range(shuffle):
            cls._shuffle()
            result -= cls._exec()

        return result

    def transform(self, trajs):
        for traj in trajs:
            yield self.partial_transform(traj)

    def __init__(self, normed=False, **kwargs):
        self.data1 = None
        self.data2 = None
        self._est = ncmi if normed else cmi

        super(TransferEntropyBase, self).__init__(**kwargs)


class AlphaAngleTransferEntropy(AlphaAngleMetricBase, TransferEntropyBase):

    """Mutual information calculations for alpha angles"""


class ContactTransferEntropy(ContactMetricBase, TransferEntropyBase):

    """Transfer entropy calculations for contacts"""


class DihedralTransferEntropy(DihedralMetricBase, TransferEntropyBase):

    """Transfer entropy calculations for dihedral angles"""
