from ..utils import floor_threshold, Timing
from ..utils import shuffle as shuffle_data

from multiprocessing import cpu_count

import pandas as pd
import numpy as np

from msmbuilder.featurizer import (AlphaAngleFeaturizer, ContactFeaturizer,
                                   DihedralFeaturizer)


class BaseMetric(object):

    """Base metric object"""

    def _shuffle(self):
        self.shuffled_data = shuffle_data(self.shuffled_data)

    def _extract_data(self, traj):
        pass

    def _before_exec(self, traj):
        self.data = self._extract_data(traj)
        self.shuffled_data = self.data
        self.labels = np.unique(self.data.columns.levels[0])

    def _exec(self):
        pass

    def _floored_exec(self):
        return floor_threshold(self._exec())

    def partial_transform(self, traj, shuffle=0, verbose=False):
        """Transform a single mdtraj.Trajectory into an array of metric scores.

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
        result : np.ndarray
            Scoring matrix
        """
        self._before_exec(traj)
        result = self._floored_exec()
        correction = np.zeros_like(result)
        for i in range(shuffle):
            with Timing(i, verbose=verbose):
                self._shuffle()
                correction += self._floored_exec()

        return floor_threshold(result - np.nan_to_num(correction / shuffle))

    def transform(self, trajs, shuffle=0, verbose=False):
        """Invokes partial_transform over a list of mdtraj.Trajectory objects

        Parameters
        ----------
        trajs : list
            List of trajectories to transform
        shuffle : int
            Number of shuffle iterations (default: 0)
        verbose : bool
            Whether to display performance

        Returns
        -------
        result : array_like
            List of scoring matrices
        """
        for traj in trajs:
            yield self.partial_transform(traj, shuffle=shuffle,
                                         verbose=verbose)

    def __init__(self, n_bins=3, rng=None, method='knn',
                 threads=None):
        self.data = None
        self.shuffled_data = None
        self.labels = None
        self.n_bins = n_bins
        self.rng = rng
        self.method = method
        self.n_threads = threads or int(cpu_count() / 2)


class DihedralBaseMetric(BaseMetric):

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
            data.append(pd.DataFrame((angles + np.pi) % (2. * np.pi),
                        columns=[idx, len(idx) * [tp]]))
        return pd.concat(data, axis=1)

    def __init__(self, types=None, rng=None, **kwargs):
        self.types = types or ['phi', 'psi']
        self.rng = rng or [0., 2 * np.pi]

        super(DihedralBaseMetric, self).__init__(**kwargs)


class AlphaAngleBaseMetric(DihedralBaseMetric):

    """Base alpha angle metric object"""

    def _featurizer(self, **kwargs):
        return AlphaAngleFeaturizer(sincos=False)

    def __init__(self, **kwargs):
        self.types = ['alpha']

        super(AlphaAngleBaseMetric, self).__init__(**kwargs)


class ContactBaseMetric(BaseMetric):

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

        super(ContactBaseMetric, self).__init__(**kwargs)
