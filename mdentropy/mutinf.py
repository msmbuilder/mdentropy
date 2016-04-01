from mdentropy.utils import dihedrals, shuffle
from mdentropy.entropy import mi, nmi

import numpy as np
from itertools import product
from itertools import combinations_with_replacement as combinations

from multiprocessing import cpu_count, Pool
from contextlib import closing


class MutualInformationBase(object):

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
                               method=cls.method)

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

    def _shuffle(cls):
        cls.data = shuffle(cls.data)

    def partial_transform(cls, traj, shuffle=False):
        cls.data = cls._extract_data(traj)
        cls.labels = np.unique(np.hstack([df.columns for df in cls.data]))
        if shuffle:
            cls.shuffle()
        return cls._mutinf()

    def transform(cls, trajs):
        for traj in trajs:
            yield cls.partial_transform(traj)

    def __init__(cls, nbins=24, method='chaowangjost', normed=False,
                 threads=None):
        cls.n_types = 1
        cls.data = None
        cls.labels = None
        cls.n_bins = nbins
        cls.method = method
        cls._est = mi
        cls.n_threads = int(cpu_count()/2)

        if normed:
            cls._est = nmi
        if threads is not None:
            cls.n_threads = threads


class DihedralMutualInformation(MutualInformationBase):

    def _extract_data(self, traj):
        return dihedrals(traj, types=self.types)

    def __init__(self, types=['phi', 'psi'], **kwargs):
        self.types = types
        self.n_types = len(types)
        super(DihedralMutualInformation, self).__init__(**kwargs)
