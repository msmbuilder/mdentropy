from .entropy import ent, ce
from ..utils import avgdigamma

from numpy import finfo, float32, log, nan_to_num, random, sqrt, vstack

from scipy.spatial import cKDTree
from scipy.special import psi

__all__ = ['mi', 'nmi', 'cmi', 'ncmi']
EPS = finfo(float32).eps


def mi(n_bins, x, y, rng=None, method='grassberger'):
    """Mutual information calculation

    Parameters
    ----------
    n_bins : int
        Number of bins.
    x : array_like, shape = (n_samples, )
        Independent variable
    y : array_like, shape = (n_samples, )
        Independent variable
    rng : list
        List of min/max values to bin data over.
    method : {'kde', 'chaowangjost', 'grassberger', None}
        Method used to calculate entropy.
    Returns
    -------
    entropy : float
    """
    if method == 'knn':
        return knn_mi(x, y, k=n_bins)

    return (ent(n_bins, [rng], method, x) +
            ent(n_bins, [rng], method, y) -
            ent(n_bins, 2 * [rng], method, x, y))


def knn_mi(x, y, k=3):
    """ Mutual information of x and y
        x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    # small noise to break degeneracy, see doc.

    x += EPS * random.rand(x.shape[0], x.shape[1])
    y += EPS * random.rand(y.shape[0], y.shape[1])
    points = vstack((x, y)).T
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = cKDTree(points)
    dvec = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in points]
    a, b, c, d = (avgdigamma(x.T, dvec), avgdigamma(y.T, dvec),
                  psi(k), psi(points.shape[0]))
    return (-a - b + c + d) / log(2)


def nmi(n_bins, x, y, rng=None, method='grassberger'):
    """Normalized mutual information calculation

    Parameters
    ----------
    n_bins : int
        Number of bins.
    x : array_like, shape = (n_samples, )
        Independent variable
    y : array_like, shape = (n_samples, )
        Independent variable
    rng : list
        List of min/max values to bin data over.
    method : {'kde', 'chaowangjost', 'grassberger', None}
        Method used to calculate entropy.
    Returns
    -------
    entropy : float
    """
    return nan_to_num(mi(n_bins, x, y, method=method, rng=rng) /
                      sqrt(ent(n_bins, [rng], method, x) *
                      ent(n_bins, [rng], method, y)))


def cmi(n_bins, x, y, z, rng=None, method='grassberger'):
    """Conditional mutual information calculation

    Parameters
    ----------
    n_bins : int
        Number of bins.
    x : array_like, shape = (n_samples, )
        Conditioned variable
    y : array_like, shape = (n_samples, )
        Conditioned variable
    z : array_like, shape = (n_samples, )
        Conditional variable
    rng : list
        List of min/max values to bin data over.
    method : {'kde', 'chaowangjost', 'grassberger', None}
        Method used to calculate entropy.
    Returns
    -------
    entropy : float
    """
    if method == 'knn':
        return knn_cmi(x, y, z, k=n_bins)

    return (ent(n_bins, 2 * [rng], method, x, z) +
            ent(n_bins, 2 * [rng], method, y, z) -
            ent(n_bins, [rng], method, z) -
            ent(n_bins, 3 * [rng], method, x, y, z))


def knn_cmi(x, y, z, k=3):
    """ Mutual information of x and y, conditioned on z
        x, y, z should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    # small noise to break degeneracy, see doc.
    x += EPS * random.rand(x.shape[0], x.shape[1])
    y += EPS * random.rand(y.shape[0], y.shape[1])
    z += EPS * random.rand(z.shape[0], z.shape[1])
    points = vstack((x, y, z)).T
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = cKDTree(points)
    dvec = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in points]
    a, b, c, d = (avgdigamma(vstack((x, z)).T, dvec),
                  avgdigamma(vstack((y, z)).T, dvec),
                  avgdigamma(z.T, dvec), psi(k))
    return (-a - b + c + d) / log(2)


def ncmi(n_bins, x, y, z, rng=None, method='grassberger'):
    """Normalized conditional mutual information calculation

    Parameters
    ----------
    n_bins : int
        Number of bins.
    x : array_like, shape = (n_samples, )
        Conditioned variable
    y : array_like, shape = (n_samples, )
        Conditioned variable
    z : array_like, shape = (n_samples, )
        Conditional variable
    rng : list
        List of min/max values to bin data over.
    method : {'kde', 'chaowangjost', 'grassberger', None}
        Method used to calculate entropy.
    Returns
    -------
    entropy : float
    """

    return (cmi(n_bins, x, y, z, rng=rng, method=method) /
            ce(n_bins, x, z, rng=rng, method=method))

    # return nan_to_num(1 + (ent(n_bins, 2 * [rng], method, y, z) -
    #                   ent(n_bins, 3 * [rng], method, x, y, z)) /
    #                   ce(n_bins, x, z, rng=rng, method=method))
