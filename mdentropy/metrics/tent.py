from ..core import cmutinf, ncmutinf
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
                         self.data[j].values,
                         self.shuffled_data[i].values,
                         self.shuffled_data[j].values,
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
        self.data = self._extract_data(traj2)
        self.shuffled_data = self._extract_data(traj1)
        self.labels = np.unique(self.data.columns.levels[0])

    def __init__(self, normed=True, **kwargs):
        self._est = ncmutinf if normed else cmutinf
        self.partial_transform.__func__.__doc__ = """
        Partial transform a mdtraj.Trajectory into an n_residue by n_residue
            matrix of transfer entropy scores.

            Parameters
            ----------
            traj : tuple
                Pair of trajectories to transform (state0, state1)
            shuffle : int
                Number of shuffle iterations (default: 0)
            verbose : bool
                Whether to display performance

            Returns
            -------
            result : np.ndarray, shape = (n_residue, n_residue)
                Transfer entropy matrix
        """

        super(TransferEntropyBase, self).__init__(**kwargs)


class AlphaAngleTransferEntropy(AlphaAngleBaseMetric, TransferEntropyBase):

    """Transfer entropy calculations for alpha angles"""


class ContactTransferEntropy(ContactBaseMetric, TransferEntropyBase):

    """Transfer entropy calculations for contacts"""


class DihedralTransferEntropy(DihedralBaseMetric, TransferEntropyBase):

    """Transfer entropy calculations for dihedral angles"""
