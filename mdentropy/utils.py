from __future__ import print_function

import time
import numpy as np
from numpy import linspace, histogramdd, product, vstack, sum
import pandas as pd
import mdtraj as md
from scipy.stats import chi2
from itertools import product as iterproduct


class timing(object):
    "Context manager for printing performance"
    def __init__(self, iter):
        self.iter = iter

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, ty, val, tb):
        end = time.time()
        print("Round %d : %0.3f seconds" %
              (self.iter, end-self.start))
        return False


def hist(nbins, r, *args):
    data = vstack((args)).T
    return histogramdd(data, bins=nbins, range=r)[0].flatten()


def adaptive(X, r=None, alpha=0.05):

    N = X.shape[1]
    dims = range(N)
    x2 = chi2.ppf(1-alpha, N-1)
    counts = []

    if r is None:
        r = tuple((X[:, i].min(), X[:, i].max()) for i in dims)

    # Estimate of X2 statistic
    def sX2(freq):
        return sum((freq - freq.mean())**2)/freq.mean()

    def dvpartition(X, r):
        nonlocal N
        nonlocal counts
        nonlocal dims
        nonlocal x2

        Y = X[product([(i[0] <= X[:, j])*(i[1] >= X[:, j])
                       for j, i in enumerate(r)], 0).astype(bool), :]

        part = [linspace(r[i][0], r[i][1], 3) for i in dims]

        partitions = set([])
        newr = set((tuple((part[i][j[i]], part[i][j[i]+1])
                    for i in dims)
                    for j in iterproduct(*(N*[[0, 1]]))),)
        freq = histogramdd(Y, bins=part)[0]
        chisq = (sX2(freq) > x2)
        if chisq and False not in ((X.max(0) - X.min(0)).T > 0):
            for nr in newr:
                newpart = dvpartition(X, r=nr)
                for newp in newpart:
                    partitions.update(tuple((newp,)))
        elif Y.shape[0] > 0:
            partitions = set(tuple((r,)))
            counts.append(Y.shape[0])
        return partitions

    return array(list(dvpartition(X, r))), array(counts).astype(int)


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
    sdf = df.copy()
    sampler = np.random.permutation
    for i in range(n):
        sdf = sdf.apply(sampler, axis=0)
        sdf = sdf.apply(sampler, axis=1)
    return sdf


class Dihedrals(object):
    def __call__(self, traj):
        atoms, angles = self.method(traj)
        idx = [traj.topology.atom(i).residue.index
               for i in atoms[:, self.type]]
        return pd.DataFrame(180*angles/np.pi, columns=idx)

    def __init__(self, method, type):
        assert type < 3 or type > -1
        self.type = type
        self.method = method


def dihedrals(traj):
    kinds = [
        Dihedrals(md.compute_phi, 2),
        Dihedrals(md.compute_psi, 1),
        Dihedrals(md.compute_chi1, 0),
        Dihedrals(md.compute_chi2, 1)
        ]
    return [kind(traj) for kind in kinds]
