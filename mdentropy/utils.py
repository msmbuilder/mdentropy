from __future__ import print_function

import time
import pandas as pd
import numpy as np
from numpy import linspace, histogramdd, product, sum, array
from scipy.stats import chi2
from itertools import product as iterproduct

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


def adaptive(*args, rng=None, min_partitions=4, alpha=None):
    """ Darbellay-Vajda adaptive partitioning
    doi:10.1109/18.761290
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
    X = np.vstack(args).T
    n_dims = X.shape[1]
    dims = range(n_dims)

    if rng is None:
        rng = tuple((X[:, i].min(), X[:, i].max()) for i in dims)

    if alpha is None:
        alpha = 1/X.shape[0]

    def sX2(freq):
        return sum((freq - freq.mean())**2)/freq.mean()

    def dvpartition(X, rng):
        nonlocal n_dims
        nonlocal counts
        nonlocal dims
        nonlocal x2

        Y = X[product([(i[0] <= X[:, j])*(i[1] >= X[:, j])
                       for j, i in enumerate(rng)], 0).astype(bool), :]

        part = [linspace(rng[i][0], rng[i][1], 3) for i in dims]

        partitions = set([])
        newrng = set((tuple((part[i][j[i]], part[i][j[i]+1]) for i in dims)
                     for j in iterproduct(*(n_dims*[[0, 1]]))),)
        freq = histogramdd(Y, bins=part)[0]
        chisq = (sX2(freq) >= x2)
        if chisq and False not in ((Y.max(0) - Y.min(0)).T > 0):
            for nr in newrng:
                newpart = dvpartition(X, rng=nr)
                for newp in newpart:
                    partitions.update(tuple((newp,)))
        elif Y.shape[0] > 0:
            partitions = set(tuple((rng,)))
            counts += (Y.shape[0],)
        return partitions
    counts = ()
    x2 = chi2.ppf(1-alpha, 2**n_dims-1)
    dvpartition(X, rng)
    return array(counts).astype(int)


def dvpartition(X, r=None, min_partitions=4, alpha=.05):

    # Estimate of X2 statistic
    def sX2(freq):
        return np.sum((freq - freq.mean())**2)/freq.mean()

    # Get number of dimensions
    N = X.shape[1]

    # If no ranges are supplied, initialize with min/max for each dimension
    # Else filter out data that is not in our initial partition
    if r is None:
        r = [[X[:, i].min(), X[:, i].max()] for i in range(N)]
    else:
        Y = X[np.product([(i[0] <= X[:, j])*(i[1] >= X[:, j])
                          for j, i in enumerate(r)], 0).astype(bool), :]

    # Subdivide our partitions by the midpoint in each dimension
    part = np.array([np.linspace(r[i][0], r[i][1], 3) for i in range(N)])
    partitions = []
    newr = [[[part[i, j[i]], part[i, j[i]+1]] for i in range(N)]
            for j in product(*(N*[[0, 1]]))]

    # Calculate counts for new partitions
    freq = np.histogramdd(Y, bins=part)[0]

    # Compare estimate to X2 statistic at given alpha
    # to determine if data in partition is not uniform.
    # Skips check if alpha is set to None.
    if alpha is not None:
        chisq = (sX2(freq) > chi2.ppf(1-alpha, N-1))
    else:
        chisq = Y.shape[0] > 0

    # If not uniform proceed
    if (chisq and np.product([Y[:, i].min() != Y[:, i].max()
                              for i in range(N)]).astype(bool)):

        # For each new partition continue dvpartition recursively
        for nr in newr:
            newpart = dvpartition(X, r=nr, alpha=alpha)
            for newp in newpart:
                partitions.append(newp)
    # If uniform and contains data, return current partition
    # unless min_partitions is not satisfied. In that case,
    # try a looser restriction on alpha.
    elif Y.shape[0] > 0:
        if min_partitions == 1:
            partitions = [r]
        while len(partitions) < min_partitions:
            for nr in newr:
                newpart = dvpartition(X, r=nr, alpha=min_partitions*alpha)
                for newp in newpart:
                    partitions.append(newp)
    return partitions


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
