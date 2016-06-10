from .entropy import ent, ce
from ..utils import avgdigamma

from numpy import (atleast_2d, diff, finfo, float32, log, nan_to_num, random,
                   sqrt, hstack)

from scipy.spatial import cKDTree
from scipy.special import psi

__all__ = ['mi', 'nmi', 'cmi', 'ncmi']
EPS = finfo(float32).eps


def mi(n_bins, x, y, rng=None, method='knn'):
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
        return knn_mi(x, y, k=n_bins,
                      boxsize=diff(rng).max() if rng else None)

    return (ent(n_bins, [rng], method, x) +
            ent(n_bins, [rng], method, y) -
            ent(n_bins, 2 * [rng], method, x, y))


def knn_mi(x, y, k=None,  boxsize=None):
    """Entropy calculation

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
    # small noise to break degeneracy, see doc.

    x += EPS * random.rand(*x.shape)
    y += EPS * random.rand(*y.shape)
    points = hstack((x, y))

    k = k if k else int(points.shape[0] * 0.01)

    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = cKDTree(points, boxsize=boxsize)
    dvec = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in points]
    a, b, c, d = (avgdigamma(atleast_2d(x).reshape(points.shape[0], -1), dvec),
                  avgdigamma(atleast_2d(y).reshape(points.shape[0], -1), dvec),
                  psi(k), psi(points.shape[0]))
    return (-a - b + c + d) / log(2)


def nmi(n_bins, x, y, rng=None, method='knn'):
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
    return nan_to_num(mi(n_bins, x, y, method=method, rng=rng) /
                      sqrt(ent(n_bins, [rng], method, x) *
                      ent(n_bins, [rng], method, y)))


def cmi(n_bins, x, y, z, rng=None, method='knn'):
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
        return knn_cmi(x, y, z, k=n_bins,
                       boxsize=diff(rng).max() if rng else None)

    return (ce(n_bins, x, z, rng=rng, method=method) +
            ent(n_bins, 2 * [rng], method, y, z) -
            ent(n_bins, 3 * [rng], method, x, y, z))


def knn_cmi(x, y, z, k=None, boxsize=None):
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
    # small noise to break degeneracy, see doc.
    x += EPS * random.rand(*x.shape)
    y += EPS * random.rand(*y.shape)
    z += EPS * random.rand(*z.shape)
    points = hstack((x, y, z))

    k = k if k else int(points.shape[0] * 0.01)

    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = cKDTree(points, boxsize=boxsize)
    dvec = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in points]
    a, b, c, d = (avgdigamma(hstack((x, z)), dvec),
                  avgdigamma(hstack((y, z)), dvec),
                  avgdigamma(atleast_2d(z).reshape(points.shape[0], -1), dvec),
                  psi(k))
    return (-a - b + c + d) / log(2)


def ncmi(n_bins, x, y, z, rng=None, method='knn'):
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

    return (cmi(n_bins, x, y, z, rng=rng, method=method) /
            ce(n_bins, x, z, rng=rng, method=method))
