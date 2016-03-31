from mdentropy.utils import dihedrals, shuffle
from mdentropy.entropy import ncmi

import numpy as np
from itertools import product


class TransferEntropyBase(object):

    def _partial_tent(cls, i, j):
        for m, n in product(range(cls.n_types),
                            range(cls.n_types)):
            if (i not in cls.data1[m].columns or
                    j not in cls.data1[n].columns):
                yield 0.0
            if i == j and m == n:
                yield 1.0
            yield ncmi(cls.n_bins,
                       cls.data2[m][j],
                       cls.data1[n][i],
                       cls.data1[m][j],
                       method=cls.method)

    def _tent(cls):

        def y(p):
            i, j = p
            return sum(cls._partial_tent(i, j))

        n = np.unique(np.hstack(tuple(map(np.array,
                                          [df.columns for df in cls.data]))))

        CMI = np.reshape(map(y, product(n, n)), (n.size, n.size)).T
        return CMI

    def _extract_data(cls, traj1, traj2):
        pass

    def _shuffle(cls):
        cls.data1 = shuffle(cls.data1)
        cls.data2 = shuffle(cls.data2)

    def partial_transform(cls, traj, shuffle=False):
        traj1, traj2 = traj
        cls.data1 = cls._extract_data(traj1)
        cls.data2 = cls._extract_data(traj2)
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


class DihedralTransferEntropy(object):

    def _extract_data(self, traj):
        return dihedrals(traj, types=self.types)

    def __init__(self, types=['phi', 'psi'], **kwargs):
        self.types = types
        self.n_types = len(types)
        super(DihedralTransferEntropy, self).__init__(**kwargs)
