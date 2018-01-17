from .entropy import entropy, centropy
from ..utils import avgdigamma, nearest_distances

from numpy import (atleast_2d, diff, finfo, float32, hstack, nan_to_num, sqrt)

from scipy.special import psi

__all__ = ['mutinf', 'nmutinf', 'cmutinf', 'ncmutinf']
EPS = finfo(float32).eps


def mutinf(n_bins, x, y, rng=None, method='knn'):
    """Mutual information calculation

    Parameters
    ----------
    n_bins : int
        Number of bins.
    x : array_like, shape = (n_samples, n_dim)
        Independent variable
    y : array_like, shape = (n_samples, n_dim)
        Independent variable
    rng : list
        List of min/max values to bin data over.
    method : {'kde', 'chaowangjost', 'grassberger', 'knn', None}
        Method used to calculate entropy.

    Returns
    -------
    entropy : float
    """
    if method == 'knn':
        return knn_mutinf(x, y, k=n_bins,
                          boxsize=diff(rng).max() if rng else None)

    return (entropy(n_bins, [rng], method, x) +
            entropy(n_bins, [rng], method, y) -
            entropy(n_bins, 2 * [rng], method, x, y))


def knn_mutinf(x, y, k=None, boxsize=None):
    """k-NN mutual information calculation

    Parameters
    ----------
    x : array_like, shape = (n_samples, n_dim)
        Independent variable
    y : array_like, shape = (n_samples, n_dim)
        Independent variable
    k : int
        Number of bins.
    boxsize : float (or None)
        Wrap space between [0., boxsize)

    Returns
    -------
    mi : float
    """
    data = hstack((x, y))

    k = k if k else max(3, int(data.shape[0] * 0.01))

    # Find nearest neighbors in joint space, p=inf means max-norm
    dvec = nearest_distances(data, k=k)
    a, b, c, d = (avgdigamma(atleast_2d(x).reshape(data.shape[0], -1), dvec),
                  avgdigamma(atleast_2d(y).reshape(data.shape[0], -1), dvec),
                  psi(k), psi(data.shape[0]))
    return max((-a - b + c + d), 0.)


def nmutinf(n_bins, x, y, rng=None, method='knn'):
    """Normalized mutual information calculation

    Parameters
    ----------
    n_bins : int
        Number of bins.
    x : array_like, shape = (n_samples, n_dim)
        Independent variable
    y : array_like, shape = (n_samples, n_dim)
        Independent variable
    rng : list
        List of min/max values to bin data over.
    method : {'kde', 'chaowangjost', 'grassberger', 'knn', None}
        Method used to calculate entropy.

    Returns
    -------
    entropy : float
    """
    return nan_to_num(mutinf(n_bins, x, y, method=method, rng=rng) /
                      sqrt(entropy(n_bins, [rng], method, x) *
                      entropy(n_bins, [rng], method, y)))


def cmutinf(n_bins, x, y, z, rng=None, method='knn'):
    """Conditional mutual information calculation

    Parameters
    ----------
    n_bins : int
        Number of bins.
    x : array_like, shape = (n_samples, n_dim)
        Conditioned variable
    y : array_like, shape = (n_samples, n_dim)
        Conditioned variable
    z : array_like, shape = (n_samples, n_dim)
        Conditional variable
    rng : list
        List of min/max values to bin data over.
    method : {'kde', 'chaowangjost', 'grassberger', 'knn', None}
        Method used to calculate entropy.

    Returns
    -------
    entropy : float
    """
    if method == 'knn':
        return knn_cmutinf(x, y, z, k=n_bins,
                           boxsize=diff(rng).max() if rng else None)

    return (centropy(n_bins, x, z, rng=rng, method=method) +
            entropy(n_bins, 2 * [rng], method, y, z) -
            entropy(n_bins, 3 * [rng], method, x, y, z))


def knn_cmutinf(x, y, z, k=None, boxsize=None):
    """Entropy calculation

    Parameters
    ----------
    x : array_like, shape = (n_samples, n_dim)
        Conditioned variable
    y : array_like, shape = (n_samples, n_dim)
        Conditioned variable
    z : array_like, shape = (n_samples, n_dim)
        Conditional variable
    k : int
        Number of bins.
    boxsize : float (or None)
        Wrap space between [0., boxsize)

    Returns
    -------
    cmi : float
    """
    data = hstack((x, y, z))

    k = k if k else max(3, int(data.shape[0] * 0.01))

    # Find nearest neighbors in joint space, p=inf means max-norm
    dvec = nearest_distances(data, k=k)
    a, b, c, d = (avgdigamma(hstack((x, z)), dvec),
                  avgdigamma(hstack((y, z)), dvec),
                  avgdigamma(atleast_2d(z).reshape(data.shape[0], -1), dvec),
                  psi(k))
    return max((-a - b + c + d), 0.)


def ncmutinf(n_bins, x, y, z, rng=None, method='knn'):
    """Normalized conditional mutual information calculation

    Parameters
    ----------
    n_bins : int
        Number of bins.
    x : array_like, shape = (n_samples, n_dim)
        Conditioned variable
    y : array_like, shape = (n_samples, n_dim)
        Conditioned variable
    z : array_like, shape = (n_samples, n_dim)
        Conditional variable
    rng : list
        List of min/max values to bin data over.
    method : {'kde', 'chaowangjost', 'grassberger', 'knn', None}
        Method used to calculate entropy.

    Returns
    -------
    ncmi : float
    """
    return (cmutinf(n_bins, x, y, z, rng=rng, method=method) /
            centropy(n_bins, x, z, rng=rng, method=method))
