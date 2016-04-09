from ..utils import shuffle

from multiprocessing import cpu_count

import pandas as pd
import numpy as np

from msmbuilder.featurizer import DihedralFeaturizer


class MetricBase(object):

    def _shuffle(cls):
        cls.data = shuffle(cls.data)

    def _extract_data(cls, traj):
        pass

    def transform(cls, trajs):
        for traj in trajs:
            yield cls.partial_transform(traj)

    def __init__(cls, n_bins=24, rng=None, method='chaowangjost',
                 threads=None):
        cls.n_types = 1
        cls.data = None
        cls.labels = None
        cls.n_bins = n_bins
        cls.rng = rng
        cls.method = method
        cls.n_threads = threads or int(cpu_count()/2)


class DihedralMetricBase(MetricBase):

    def _extract_data(cls, traj):
        data = []
        for tp in cls.types:
            featurizer = DihedralFeaturizer(types=[tp], sincos=False)
            angles = featurizer.partial_transform(traj)
            summary = featurizer.describe_features(traj)
            idx = [[traj.topology.atom(ati).residue.index
                    for ati in item['atominds']][1] for item in summary]
            data.append(pd.DataFrame(180.*angles/np.pi,
                                     columns=[idx, len(idx)*[tp]]))
        return pd.concat(data, axis=1)

    def __init__(self, types=None, **kwargs):
        self.types = types or ['phi', 'psi']
        self.n_types = len(self.types)

        super(DihedralMetricBase, self).__init__(**kwargs)
