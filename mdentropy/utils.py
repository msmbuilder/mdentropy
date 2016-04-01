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


def hist(nbins, rng, *args):
    data = np.vstack((args)).T
    return np.histogramdd(data, bins=nbins, range=rng)[0].flatten()


def shuffle(df, n=1):
    sdf = df.copy()
    sampler = np.random.permutation
    for _ in range(n):
        sdf = sdf.apply(sampler, axis=0)
        sdf = sdf.apply(sampler, axis=1)
    return sdf


class Dihedrals(object):
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
    types = types or ['phi', 'psi']
    return [Dihedrals(tp)(traj) for tp in types]
