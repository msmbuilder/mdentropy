from __future__ import print_function

import time

from numpy import dtype, random, unique, void


__all__ = ['floor_threshold', 'shuffle', 'Timing', 'unique_row_count']


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
    a : numpy.ndarray
    Returns
    -------
    counts : array_like, shape = (n_bins, )
        unique row counts
    """
    _, counts = unique(arr.view(dtype((void, arr.dtype.itemsize *
                                       arr.shape[1]))), return_counts=True)
    return counts


def floor_threshold(arr, threshold=0.):
    new_arr = arr.copy()
    new_arr[arr < threshold] = threshold
    return new_arr
