from __future__ import print_function

import time
import numpy as np
import pandas as pd
import mdtraj as md
from scipy.stats import chi2
from itertools import product


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
    X = np.vstack(args).T
    N = X.shape[0]
    parts = dvpartition(*args, r=r, alpha=1/N)
    return np.array([np.histogramdd(X, part)[0] for part in parts]).flatten()


def dvpartition(X, r=None, alpha=.05):
    # Adapted from:
    # Darbellay AG, Vajda I: Estimation of the information by an adaptive
    # partitioning of the observation space.
    # IEEE Transactions on Information Theory 1999, 45(4):1315–1321.
    # 10.1109/18.761290

    def sX2(freq):
        return np.sum((freq - freq.mean())**2)/freq.mean()

    N = X.shape[1]

    if r is None:
        r = [[X[:, i].min(), X[:, i].max()] for i in range(N)]
    Y = X[np.product([(i[0] <= X[:, j])*(i[1] >= X[:, j])
                      for j, i in enumerate(r)], 0).astype(bool), :]
    part = np.array([np.linspace(r[i][0], r[i][1], 3) for i in range(N)])
    partitions = []
    freq = np.histogramdd(Y, bins=part)[0]
    if ((sX2(freq) > chi2.ppf(1-alpha, N-1)) and
        np.product([Y[:, i].min() != Y[:, i].max()
                    for i in range(N)]).astype(bool)):
        newr = [[[part[i, j[i]], part[i, j[i]+1]] for i in range(N)]
                for j in product(range(N), range(N))]
        for nr in newr:
            newpart = dvpartition(Y, r=nr, alpha=alpha)
            for newp in newpart:
                partitions.append(newp)
    elif Y.shape[0] > 0:
        partitions = [r]
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
