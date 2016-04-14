from .entropy import ent, ce
import numpy as np

__all__ = ['mi', 'nmi', 'cmi', 'ncmi']


def mi(nbins, x, y, rng=None, method='grassberger'):
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
    return (ent(nbins, [rng], method, x) +
            ent(nbins, [rng], method, y) -
            ent(nbins, 2 * [rng], method, x, y))


def nmi(nbins, x, y, rng=None, method='grassberger'):
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
    return np.nan_to_num(mi(nbins, x, y, method=method, rng=rng) /
                         np.sqrt(ent(nbins, [rng], method, x) *
                         ent(nbins, [rng], method, y)))


def cmi(nbins, x, y, z, rng=None, method='grassberger'):
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
    return (ent(nbins, 2 * [rng], method, x, z) +
            ent(nbins, 2 * [rng], method, y, z) -
            ent(nbins, [rng], method, z) -
            ent(nbins, 3 * [rng], method, x, y, z))


def ncmi(nbins, x, y, z, rng=None, method='grassberger'):
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
    return np.nan_to_num(1 + (ent(nbins, 2 * [rng], method, y, z) -
                         ent(nbins, 3 * [rng], method, x, y, z)) /
                         ce(nbins, x, z, rng=rng, method=method))
