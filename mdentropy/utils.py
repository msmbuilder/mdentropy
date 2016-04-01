from __future__ import print_function

import time
import numpy as np
import pandas as pd
from msmbuilder.featurizer import DihedralFeaturizer


class timing(object):
    "Context manager for printing performance"
    def __init__(self, iteration):
        self.iteration = iteration

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, ty, val, tb):
        end = time.time()
        print("Round %d : %0.3f seconds" %
              (self.iteration, end-self.start))
        return False


def hist(n_bins, rng, *args):
    """Convenience function for histogramming N-dimentional data

    Parameters
    ----------
    n_bins : int
        Number of bins.
    rng : list of lists
        List of min/max values to bin data over.
    args : array_like, shape = (n_samples, )
        Data of which to histogram.
    Returns
    -------
    bins : array_like, shape = (n_bins, )
    """
    data = np.vstack((args)).T
    return np.histogramdd(data, bins=n_bins, range=rng)[0].flatten()


def shuffle(df, n=1):
    """Convenience function for shuffling values in DataFrame objects

    Parameters
    ----------
    df : pandas.DataFrame
        pandas DataFrame
    n : int
        Number of shuffling iterations.
    Returns
    -------
    sdf : array_like, shape = (n_bins, )
        shuffled DataFrame
    """
    sdf = df.copy()
    sampler = np.random.permutation
    for _ in range(n):
        sdf = sdf.apply(sampler, axis=0)
        sdf = sdf.apply(sampler, axis=1)
    return sdf


class Dihedrals(object):
    """Convenience class for extracting dihedral angle data as a DataFrame"""
    def __call__(self, traj):
        featurizer = DihedralFeaturizer(types=[self.type], sincos=False)
        angles = featurizer.partial_transform(traj)
        summary = featurizer.describe_features(traj)

        idx = [[traj.topology.atom(ati).residue.index
                for ati in item['atominds']][1] for item in summary]

        return pd.DataFrame(180.*angles/np.pi, columns=idx)

    def __init__(self, tp):
        self.type = tp


def dihedrals(traj, types=None):
    """Convenience function for extracting dihedral angle data as a list of
    DataFrame objects

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Trajectory
    types : list
        Types of dihedral data to extract (default: ['phi', 'psi']).
    Returns
    -------
    dihedrals : list
    """
    types = types or ['phi', 'psi']
    return [Dihedrals(tp)(traj) for tp in types]
