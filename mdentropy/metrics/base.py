from ..utils import shuffle

from multiprocessing import cpu_count

import pandas as pd
import numpy as np

from msmbuilder.featurizer import (AlphaAngleFeaturizer, ContactFeaturizer,
                                   DihedralFeaturizer)


class MetricBase(object):
    """Base metric object"""

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
    """Base dihedral metric object"""

    def _featurizer(self, **kwargs):
        return DihedralFeaturizer(sincos=False, **kwargs)

    def _extract_data(self, traj):
        data = []
        for tp in self.types:
            featurizer = self._featurizer(types=[tp])
            angles = featurizer.partial_transform(traj)
            summary = featurizer.describe_features(traj)
            idx = [[traj.topology.atom(ati).residue.index
                    for ati in item['atominds']][1] for item in summary]
            data.append(pd.DataFrame(180. * angles / np.pi,
                                     columns=[idx, len(idx) * [tp]]))
        return pd.concat(data, axis=1)

    def __init__(self, types=None, **kwargs):
        self.types = types or ['phi', 'psi']
        self.n_types = len(self.types)

        super(DihedralMetricBase, self).__init__(**kwargs)


class AlphaAngleMetricBase(DihedralMetricBase):
    """Base alpha angle metric object"""

    def _featurizer(self, **kwargs):
        return AlphaAngleFeaturizer(sincos=False)

    def __init__(self, **kwargs):
        self.types = ['alpha']

        super(AlphaAngleMetricBase, self).__init__(**kwargs)


class ContactMetricBase(MetricBase):
    """Base contact metric object"""

    def _extract_data(self, traj):
        contact = ContactFeaturizer(contacts=self.contacts, scheme=self.scheme,
                                    ignore_nonprotein=self.ignore_nonprotein)
        distances = contact.partial_transform(traj)
        summary = contact.describe_features(traj)
        pairs = [item['resids'] for item in summary]
        resids = np.unique(pairs)
        data = []
        for resid in resids:
            idx = list(list(set(pair) - {resid})[0]
                       for pair in pairs if resid in pair)
            mapping = np.array([True if resid in pair else False
                                for pair in pairs])
            data.append(pd.DataFrame(distances[:, mapping],
                        columns=[idx, len(idx) * [resid]]))

        return pd.concat(data, axis=1)

    def __init__(self, contacts='all', scheme='closest-heavy',
                 ignore_nonprotein=True, **kwargs):
        self.contacts = contacts
        self.scheme = scheme
        self.ignore_nonprotein = ignore_nonprotein

        super(ContactMetricBase, self).__init__(**kwargs)
