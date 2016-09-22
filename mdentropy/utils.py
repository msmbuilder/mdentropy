from __future__ import print_function

import time
from glob import glob
from itertools import chain

from numpy import (dtype, exp, finfo, float32, isscalar, linspace, log,
                   meshgrid, nan_to_num, pi, random, ravel, unique, void,
                   vstack)
from numpy.linalg import det

from sklearn.neighbors import KernelDensity, NearestNeighbors, BallTree
from scipy.special import digamma, gamma


__all__ = ['floor_threshold', 'parse_files', 'shuffle', 'Timing',
           'unique_row_count', 'kde', 'nearest_distances', 'avgdigamma',
           'volume_unit_ball']
EPS = finfo(float32).eps


class Timing(object):
    "Context manager for printing performance"
    def __init__(self, iteration, verbose=False):
        self.iteration = iteration
        self.start = None
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, ty, val, tb):
        end = time.time()
        if self.verbose:
            print("Round %d : %0.3f seconds" %
                  (self.iteration, end - self.start))
        return False


def parse_files(expr):
    expr = expr.replace(' ', '').split(',')
    return list(chain(*map(glob, expr)))


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
    sampler = random.permutation
    for _ in range(n):
        sdf = sdf.apply(sampler, axis=0)
        sdf = sdf.apply(sampler, axis=1)
    return sdf


def unique_row_count(arr):
    """Convenience function for counting unique rows in a numpy.ndarray

    Parameters
    ----------
    arr : numpy.ndarray
    Returns
    -------
    counts : array_like, shape = (n_bins, )
        unique row counts
    """
    _, counts = unique(arr.view(dtype((void, arr.dtype.itemsize *
                                       arr.shape[1]))), return_counts=True)
    return counts


def floor_threshold(arr, threshold=0.):
    """Convenience funtion for thresholding to a lower bound

    Parameters
    ----------
    arr : numpy.ndarray
    Returns
    -------
    new_arr : numpy.ndarray
        thresholded array
    """
    new_arr = nan_to_num(arr.copy())
    new_arr[arr < threshold] = threshold
    return new_arr


def entropy_gaussian(C):
    '''
    Entropy of a gaussian variable with covariance matrix C
    '''
    if isscalar(C):
        return .5 * (1 + log(2 * pi)) + .5 * log(C)
    else:
        n = C.shape[0]
        return .5 * n * (1 + log(2 * pi)) + .5 * log(abs(det(C)))


def kde(data, rng, grid_size=10,  **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    n_samples = data.shape[0]
    n_dims = data.shape[1]

    bandwidth = (n_samples * (n_dims + 2) / 4.)**(-1. / (n_dims + 4.))
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(data)

    space = [linspace(i[0], i[1], grid_size) for i in rng]
    grid = meshgrid(*tuple(space))

    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(vstack(map(ravel, grid)).T)
    return exp(log_pdf), space


def nearest_distances(X, k=1, leaf_size=16):
    '''
    X = array(N,M)
    N = number of points
    M = number of dimensions
    returns the distance to the kth nearest neighbor for every point in X
    '''
    # small amount of noise to break degeneracy.
    X += EPS * random.rand(*X.shape)

    knn = NearestNeighbors(n_neighbors=k + 1, leaf_size=leaf_size,
                           p=float('inf'))
    knn.fit(X)
    d, _ = knn.kneighbors(X)  # the first nearest neighbor is itself
    return d[:, -1]


def avgdigamma(data, dvec, leaf_size=16):
    """Convenience function for finding expectation value of <psi(nx)> given
    some number of neighbors in some radius in a marginal space.

    Parameters
    ----------
    points : numpy.ndarray
    dvec : array_like (n_points,)
    Returns
    -------
    avgdigamma : float
        expectation value of <psi(nx)>
    """
    tree = BallTree(data, leaf_size=leaf_size, p=float('inf'))

    n_points = tree.query_radius(data, dvec - EPS, count_only=True)

    return digamma(n_points).mean()


def volume_unit_ball(n_dims):
    return (pi ** (.5 * n_dims)) / gamma(.5 * n_dims + 1)
