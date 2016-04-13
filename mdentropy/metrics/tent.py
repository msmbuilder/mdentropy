from .base import (AlphaAngleMetricBase, ContactMetricBase, DihedralMetricBase,
                   MetricBase)
from ..utils import shuffle
from ..core import cmi, ncmi

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

    def _tent(self):

        with closing(Pool(processes=self.n_threads)) as pool:
            CMI = list(pool.map(self._partial_tent,
                                product(self.labels, self.labels)))
            pool.terminate()

        return np.reshape(CMI, (self.labels.size, self.labels.size)).T

    def _shuffle(self):
        self.data2 = shuffle(self.data2)

    def partial_transform(self, traj, shuffled=False):
        traj1, traj2 = traj
        self.data1 = self._extract_data(traj1)
        self.data2 = self._extract_data(traj2)
        self.labels = np.unique(self.data1.columns.levels[0])
        if shuffled:
            self._shuffle()
        else:
            self.shuffled_data = self.data2

        return self._tent()

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
