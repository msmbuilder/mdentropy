from .base import MetricBase
from ..utils import dihedrals, shuffle
from ..core import cmi, ncmi

import numpy as np
from itertools import product

from multiprocessing import Pool
from contextlib import closing


class TransferEntropyBase(MetricBase):

    def _partial_tent(cls, p):
        i, j = p

        def y(i, j):
            for m, n in product(range(cls.n_types),
                                range(cls.n_types)):
                if (i not in cls.data1[m].columns or
                        j not in cls.data1[n].columns):
                    yield 0.0
                if i == j and m == n:
                    yield 1.0
                yield cls._est(cls.n_bins,
                               cls.data2[m][j],
                               cls.data1[n][i],
                               cls.data1[m][j],
                               range=cls.range,
                               method=cls.method)

        return sum(y(i, j))

    def _tent(cls):

        with closing(Pool(processes=cls.n_threads)) as pool:
            CMI = list(pool.map(cls._partial_tent,
                                product(cls.labels, cls.labels)))
            pool.terminate()

        return np.reshape(CMI, (cls.labels.size, cls.labels.size)).T

    def _extract_data(cls, traj1, traj2):
        pass

    def _shuffle(cls):
        cls.data1 = shuffle(cls.data1)
        cls.data2 = shuffle(cls.data2)

    def partial_transform(cls, traj, shuffle=False):
        traj1, traj2 = traj
        cls.data1 = cls._extract_data(traj1)
        cls.data2 = cls._extract_data(traj2)
        cls.labels = np.unique(np.hstack([df.columns for df in cls.data1]))
        if shuffle:
            cls.shuffle()
        return cls._tent()

    def transform(cls, trajs):
        for traj in trajs:
            yield cls.partial_transform(traj)

    def __init__(cls, normed=False, **kwargs):
        cls._est = ncmi if normed else cmi

        super(TransferEntropyBase, cls).__init__(**kwargs)


class DihedralTransferEntropy(TransferEntropyBase):

    def _extract_data(self, traj):
        return dihedrals(traj, types=self.types)

    def __init__(self, types=None, **kwargs):
        self.types = types or ['phi', 'psi']
        self.n_types = len(self.types)

        super(DihedralTransferEntropy, self).__init__(**kwargs)
