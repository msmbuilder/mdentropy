from ..core import ent

import numpy as np
from numpy.testing import assert_allclose as eq

COV = np.array([[1., .3], [.3, 1.]])
TRUE_ENTROPY = .5 * (COV.shape[0] * (1. + np.log(2. * np.pi)) +
                     np.linalg.det(COV))

a, b = np.random.multivariate_normal([0, 0], COV, size=1000).T
RNG = ((a.min(), a.max()), (b.min(), b.max()))


def test_kde():
    eq(ent(8, RNG, 'kde', a, b), TRUE_ENTROPY, rtol=.4)


def test_knn():
    eq(ent(3, RNG, 'knn', a, b), TRUE_ENTROPY, rtol=.4)


def test_chaowangjost():
    eq(ent(8, RNG, 'chaowangjost', a, b), TRUE_ENTROPY, rtol=.2)


def test_grassberger():
    eq(ent(8, RNG, 'grassberger', a, b), TRUE_ENTROPY, rtol=.2)


def test_adaptive():
    eq(ent(None, RNG, 'grassberger', a, b), TRUE_ENTROPY, rtol=.4)


def test_naive():
    eq(ent(8, RNG, None, a, b), TRUE_ENTROPY, rtol=.2)
