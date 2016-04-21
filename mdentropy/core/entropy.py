from .binning import symbolic

from itertools import chain

from numpy import ndarray
from numpy import sum as npsum
from numpy import (arange, bincount, diff, linspace, log, log2,
                   meshgrid, nan_to_num, nansum, product,
                   random, ravel, split, vstack, exp)

from scipy.spatial import cKDTree
from scipy.stats import entropy as naive
from scipy.special import psi


from sklearn.neighbors import KernelDensity

__all__ = ['ent', 'ce']


def ent(n_bins, rng, method, *args):
    """Entropy calculation

    Parameters
    ----------
    n_bins : int
        Number of bins.
    rng : list of lists
        List of min/max values to bin data over.
    method : {'kde', 'chaowangjost', 'grassberger', None}
        Method used to calculate entropy.
    args : numpy.ndarray, shape = (n_samples, ) or (n_features, n_samples)
        Data of which to calculate entropy. Each array must have the same
        number of samples.
    Returns
    -------
    entropy : float
    """
    args = list(chain(*[map(ndarray.flatten, split(arg, arg.shape[0]))
                        if arg.ndim == 2
                        else [arg]
                        for arg in args]))

    if rng is None or None in rng:
        rng = len(args)*[None]

    for i, arg in enumerate(args):
        if rng[i] is None:
            rng[i] = (min(arg), max(arg))

    if method == 'kde':
        return kde_ent(rng, *args, grid_size=n_bins or 20)
    if method == 'knn':
        return knn_ent(*args, k=n_bins or 3)

    counts = symbolic(n_bins, rng, *args)

    if method == 'chaowangjost':
        return chaowangjost(counts)
    elif method == 'grassberger':
        return grassberger(counts)

    return naive(counts)


def ce(n_bins, x, y, rng=None, method='grassberger'):
    """Condtional entropy calculation

    Parameters
    ----------
    n_bins : int
        Number of bins.
    x : array_like, shape = (n_samples, )
        Conditioned variable.
    y : array_like, shape = (n_samples, )
        Conditional variable.
    rng : list
        List of min/max values to bin data over.
    method : {'kde', 'chaowangjost', 'grassberger', None}
        Method used to calculate entropy.
    Returns
    -------
    entropy : float
    """
    return (ent(n_bins, 2 * [rng], method, x, y) -
            ent(n_bins, [rng], method, y))


def knn_ent(*args, k=3):
    """ The classic K-L k-nearest neighbor continuous entropy estimator
        x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    data = vstack((args)).T
    n_samples = data.shape[0]
    n_dims = data.shape[1]

    intens = 1e-6  # small noise to break degeneracy, see doc.
    data = [pt + intens * random.rand(n_dims) for pt in data]
    tree = cKDTree(data)
    nn = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in data]
    const = psi(n_samples) - psi(k) + n_dims * log(2)
    return (const + n_dims * log(nn).mean())/log(2)


def kde_ent(rng, *args, grid_size=20, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    data = vstack((args)).T
    n_samples = data.shape[0]
    n_dims = data.shape[1]

    bandwidth = (n_samples * (n_dims + 2) / 4.)**(-1. / (n_dims + 4.))
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(data)

    space = [linspace(i[0], i[1], grid_size) for i in rng]
    grid = meshgrid(*tuple(space))

    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(vstack(map(ravel, grid)).T)
    pr = exp(log_pdf)

    return -nansum(pr * log2(pr)) * product(diff(space)[:, 0])


def grassberger(counts):
    """Entropy calculation using Grassberger correction.
    doi:10.1016/0375-9601(88)90193-4

    Parameters
    ----------
    counts : list
        bin counts
    Returns
    -------
    entropy : float
    """
    n_samples = npsum(counts)
    return npsum(counts * (log(n_samples) -
                           nan_to_num(psi(counts)) -
                           ((-1.) ** counts / (counts + 1.)))) / n_samples


def chaowangjost(counts):
    """Entropy calculation using Chao, Wang, Jost correction.
    doi: 10.1111/2041-210X.12108

    Parameters
    ----------
    counts : list
        bin counts
    Returns
    -------
    entropy : float
    """
    n_samples = npsum(counts)
    bcbc = bincount(counts.astype(int))
    if len(bcbc) < 3:
        return grassberger(counts)
    if bcbc[2] == 0:
        if bcbc[1] == 0:
            A = 1.
        else:
            A = 2. / ((n_samples - 1.) * (bcbc[1] - 1.) + 2.)
    else:
        A = 2. * bcbc[2] / ((n_samples - 1.) * (bcbc[1] - 1.) +
                            2. * bcbc[2])
    pr = arange(1, int(n_samples))
    pr = 1. / pr * (1. - A) ** pr
    entropy = npsum(counts / n_samples * (psi(n_samples) -
                    nan_to_num(psi(counts))))

    if bcbc[1] > 0 and A != 1.:
        entropy += nan_to_num(bcbc[1] / n_samples *
                              (1 - A) ** (1 - n_samples *
                                          (-log(A) - npsum(pr))))
    return entropy
