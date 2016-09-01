from ..core import entropy
from ..utils import entropy_gaussian

import numpy as np
from numpy.testing import assert_allclose as eq
from unittest import skip

COV = np.array([[1., .3], [.3, 1.]])
TRUE_ENTROPY = entropy_gaussian(COV)

rs = np.random.RandomState(42)
a, b = rs.multivariate_normal([0, 0], COV, size=1000).T
RNG = ((a.min() - 0.1, a.max() + 0.1),
       (b.min() - 0.1, b.max() + 0.1))


def test_kde():
    eq(entropy(8, RNG, 'kde', a, b), TRUE_ENTROPY, rtol=.4)


def test_knn():
    eq(entropy(3, [None], 'knn', a, b), TRUE_ENTROPY, rtol=.2)


def test_chaowangjost():
    eq(entropy(8, RNG, 'chaowangjost', a, b), TRUE_ENTROPY, rtol=.2)


def test_grassberger():
    eq(entropy(8, RNG, 'grassberger', a, b), TRUE_ENTROPY, rtol=.2)


@skip('adaptive is still experimental')
def test_adaptive():
    eq(entropy(None, RNG, 'grassberger', a, b), TRUE_ENTROPY, rtol=.4)


def test_naive():
    eq(entropy(8, RNG, None, a, b), TRUE_ENTROPY, rtol=.2)
