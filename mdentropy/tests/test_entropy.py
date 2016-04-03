from ..core import ent

import numpy as np
from numpy.testing import assert_almost_equal as eq

COV = np.array([[1, .3], [.3, 1]])
TRUE_ENTROPY = .5*COV.shape[0]*(1+np.log(2*np.pi)) + .5*np.linalg.det(COV)


def test_kde():
    a, b = np.random.multivariate_normal([0, 0], COV, size=1000).T
    eq(ent(24, None, 'kde', a, b), TRUE_ENTROPY, 3)


def test_chaowangjost():
    a, b = np.random.multivariate_normal([0, 0], COV, size=1000).T
    eq(ent(24, None, 'chaowangjost', a, b), TRUE_ENTROPY, 3)


def test_grassberger():
    a, b = np.random.multivariate_normal([0, 0], COV, size=1000).T
    eq(ent(24, None, 'grassberger', a, b), TRUE_ENTROPY, 3)


def test_naive():
    a, b = np.random.multivariate_normal([0, 0], COV, size=1000).T
    eq(ent(24, None, None, a, b), TRUE_ENTROPY, 3)


def test_adaptive():
    a, b = np.random.multivariate_normal([0, 0], COV, size=1000).T
    eq(ent(None, None, 'grassberger', a, b), TRUE_ENTROPY, 3)
