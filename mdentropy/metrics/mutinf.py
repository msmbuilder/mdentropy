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
        M = np.zeros((self.labels.size, self.labels.size))
        triu = np.zeros((int((self.labels.size ** 2 + self.labels.size) / 2),))

        with closing(Pool(processes=self.n_threads)) as pool:
            values = pool.map(self._partial_mutinf,
                              combinations(self.labels, 2))
            pool.terminate()

        for i, value in enumerate(values):
            triu[i] = value

        idx = np.triu_indices_from(M)
        M[idx] = triu

        return M + M.T - np.diag(M.diagonal())

    def __init__(self, normed=True, **kwargs):
        self._est = nmi if normed else mi
        self.partial_transform.__func__.__doc__ = """
        Partial transform a mdtraj.Trajectory into an n_residue by n_residue
            matrix of mutual information scores.

            Parameters
            ----------
            traj : mdtraj.Trajectory
                Trajectory to transform
            shuffle : int
                Number of shuffle iterations (default: 0)
            verbose : bool
                Whether to display performance
            Returns
            -------
            result : np.ndarray, shape = (n_residue, n_residue)
                Mutual information matrix
        """

        super(MutualInformationBase, self).__init__(**kwargs)


class AlphaAngleMutualInformation(AlphaAngleBaseMetric, MutualInformationBase):

    """Mutual information calculations for alpha angles"""


class ContactMutualInformation(ContactBaseMetric, MutualInformationBase):

    """Mutual information calculations for contacts"""


class DihedralMutualInformation(DihedralBaseMetric, MutualInformationBase):

    """Mutual information calculations for dihedral angles"""
