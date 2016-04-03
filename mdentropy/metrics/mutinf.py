from .base import MetricBase
from ..utils import dihedrals
from ..core import mi, nmi

import numpy as np
from itertools import product
from itertools import combinations_with_replacement as combinations

from multiprocessing import Pool
from contextlib import closing


class MutualInformationBase(MetricBase):
    """
    Base mutual information object
    """

    def _partial_mutinf(cls, p):
        i, j = p

        def y(i, j):
            for m, n in product(range(cls.n_types),
                                range(cls.n_types)):
                if (i not in cls.data[m].columns or
                        j not in cls.data[n].columns):
                    yield 0.0
                if i == j and m == n:
                    yield 1.0
                yield cls._est(cls.n_bins, cls.data[m][i], cls.data[n][j],
                               rng=cls.rng, method=cls.method)

        return sum(y(i, j))

    def _mutinf(cls):
        idx = np.triu_indices(cls.labels.size)
        M = np.zeros((cls.labels.size, cls.labels.size))

        with closing(Pool(processes=cls.n_threads)) as pool:
            M[idx] = list(pool.map(cls._partial_mutinf,
                                   combinations(cls.labels, 2)))
            pool.terminate()

        M[idx[::-1]] = M[idx]

        return M

    def _extract_data(cls, traj):
        pass

    def partial_transform(cls, traj, shuffled=False):
        cls.data = cls._extract_data(traj)
        cls.labels = np.unique(np.hstack([df.columns for df in cls.data]))
        if shuffled:
            cls.shuffle()
        return cls._mutinf()

    def __init__(cls, normed=False, **kwargs):
        cls.data = None
        cls._est = nmi if normed else mi

        super(MutualInformationBase, cls).__init__(**kwargs)


class DihedralMutualInformation(MutualInformationBase):
    """
    Mutual information calculations for dihedral angles
    """

    def _extract_data(self, traj):
        return dihedrals(traj, types=self.types)

    def __init__(self, types=None, **kwargs):
        self.types = types or ['phi', 'psi']
        self.n_types = len(self.types)

        super(DihedralMutualInformation, self).__init__(**kwargs)
