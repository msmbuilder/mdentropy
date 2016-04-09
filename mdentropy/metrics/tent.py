from .base import MetricBase, DihedralMetricBase
from ..utils import shuffle
from ..core import cmi, ncmi

import numpy as np
from itertools import product

from multiprocessing import Pool
from contextlib import closing


class TransferEntropyBase(MetricBase):
    """
    Base transfer entropy object
    """

    def _partial_tent(cls, p):
        i, j = p

        return cls._est(cls.n_bins,
                        cls.data2[j].values.T,
                        cls.data1[i].values.T,
                        cls.data1[j].values.T,
                        rng=cls.rng,
                        method=cls.method)

    def _tent(cls):

        with closing(Pool(processes=cls.n_threads)) as pool:
            CMI = list(pool.map(cls._partial_tent,
                                product(cls.labels, cls.labels)))
            pool.terminate()

        return np.reshape(CMI, (cls.labels.size, cls.labels.size)).T

    def _shuffle(cls):
        cls.data1 = shuffle(cls.data1)
        cls.data2 = shuffle(cls.data2)

    def partial_transform(cls, traj, shuffled=False):
        traj1, traj2 = traj
        cls.data1 = cls._extract_data(traj1)
        cls.data2 = cls._extract_data(traj2)
        cls.labels = np.unique(cls.data1.columns.levels[0])
        if shuffled:
            cls._shuffle()

        return cls._tent()

    def transform(cls, trajs):
        for traj in trajs:
            yield cls.partial_transform(traj)

    def __init__(cls, normed=False, **kwargs):
        cls.data1 = None
        cls.data2 = None
        cls._est = ncmi if normed else cmi

        super(TransferEntropyBase, cls).__init__(**kwargs)


class DihedralTransferEntropy(DihedralMetricBase, TransferEntropyBase):
    """
    Transfer entropy calculations for dihedral angles
    """
