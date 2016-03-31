from mdentropy.utils import dihedrals, shuffle
from mdentropy.entropy import nmi

import numpy as np
from itertools import product
from itertools import combinations_with_replacement as combinations


class MutualInformationBase(object):

    def _partial_mutinf(cls, i, j):
        for m, n in product(range(cls.n_types),
                            range(cls.n_types)):
            if (i not in cls.data[m].columns or j not in cls.data[n].columns):
                yield 0.0
            if i == j and m == n:
                yield 1.0
            yield nmi(cls.n_bins, cls.data[m][i], cls.data[n][j],
                      method=cls.method)

    def _mutinf(cls):

        def y(p):
            i, j = p
            return sum(cls._partial_mutinf(i, j))

        n = np.unique(np.hstack(tuple(map(np.array,
                                          [df.columns for df in cls.data]))))

        idx = np.triu_indices(n.size)
        M = np.zeros((n.size, n.size))
        M[idx] = list(map(y, combinations(n, 2)))
        M[idx[::-1]] = M[idx]
        return M

    def _extract_data(cls, traj):
        pass

    def _shuffle(cls):
        cls.data = shuffle(cls.data)

    def partial_transform(cls, traj, shuffle=False):
        cls.data = cls._extract_data(traj)
        if shuffle:
            cls.shuffle()
        return cls._mutinf()

    def transform(cls, trajs):
        for traj in trajs:
            yield cls.partial_transform(traj)

    def __init__(cls, nbins=24, method='chaowangjost'):
        cls.n_types = 1
        cls.n_bins = nbins
        cls.method = method


class DihedralMutualInformation(MutualInformationBase):

    def _extract_data(self, traj):
        return dihedrals(traj, types=self.types)

    def __init__(self, types=['phi', 'psi'], **kwargs):
        self.types = types
        self.n_types = len(types)
        super(DihedralMutualInformation, self).__init__(**kwargs)
