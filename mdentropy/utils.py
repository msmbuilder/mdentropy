from __future__ import print_function

import time

from numpy import dtype, finfo, float32, nan_to_num, random, unique, void

from scipy.spatial import cKDTree
from scipy.special import digamma


__all__ = ['floor_threshold', 'shuffle', 'Timing', 'unique_row_count',
           'avgdigamma']
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


def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    n_samples = points.shape[0]
    tree = cKDTree(points)
    avg = 0.
    for i in range(n_samples):
        dist = dvec[i]
        # subtlety, we don't include the boundary point,
        # but we are implicitly adding 1 to kraskov def bc center point is included
        n_points = len(tree.query_ball_point(points[i], dist - EPS,
                       p=float('inf')))
        avg += digamma(n_points) / n_samples
    return avg
