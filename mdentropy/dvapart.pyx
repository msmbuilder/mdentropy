from scipy.stats import chi2
from itertools import product as iterproduct
import numpy as np
from numpy import linspace, histogramdd, product, vstack

from cpython cimport bool
cimport cython

def adaptive(*args, r=None, alpha=0.05):
    # Stack data
    X = vstack(args).T

    # Get number of dimensions
    N = X.shape[1]
    dims = range(N)

    # Compute X2 statistic at given CI with Bonferoni correction
    x2 = chi2.ppf(1-alpha/4., N-1)
    counts = []

    # Estimate of X2 statistic
    def sX2(freq):
        return np.sum((freq - freq.mean())**2)/freq.mean()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef dvpartition(np.ndarray X, float[2, ::1] r):
        # Adapted from:
        # Darbellay AG, Vajda I: Estimation of the information by an adaptive
        # partitioning of the observation space.
        # IEEE Transactions on Information Theory 1999, 45(4):1315–1321.
        # 10.1109/18.761290
        cdef nonlocal int N
        cdef nonlocal int[::1] counts
        cdef nonlocal int[::1] dims
        cdef nonlocal float x2

        # If no ranges are supplied, initialize with min/max for each dimension
        # Else filter out data that is not in our initial partition
        if r is None:
            r = [[X[:, i].min(), X[:, i].max()] for i in dims]
        else:
            cdef np.ndarray  Y = X[product([(i[0] <= X[:, j])*(i[1] >= X[:, j])
                                   for j, i in enumerate(r)], 0).astype(bool),
                                   :]

        cdef float[2, ::1] part = [linspace(r[i][0], r[i][1], 3) for i in dims]

        cdef float[2, ::1] partitions = []
        cdef float[3, ::1] newr = [[[part[i][j[i]], part[i][j[i]+1]]
                                    for i in dims]
                                   for j in iterproduct(*(N*[[0, 1]]))]
        cdef np.ndarray freq = histogramdd(Y, bins=part)[0]
        cdef bool chisq = (sX2(freq) > x2)
        if chisq and False not in ((X.max(0) - X.min(0)).T > 0):
            for nr in newr:
                cdef float[2, ::1] newpart = dvpartition(X, r=nr)
                for newp in newpart:
                        partitions.append(newp)
        elif Y.shape[0] > 0:
            partitions = [r]
            counts.append(Y.shape[0])
        return partitions

    return array(dvpartition(X, r)), array(counts).astype(int)
