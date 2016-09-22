from ..utils import entropy_gaussian
from ..core import entropy

import numpy as np
from numpy.testing import assert_allclose as close

rs = np.random.RandomState(42)
n, d = 50000, 3
P = np.array([[1, 0, 0], [0, 1, .5], [0, 0, 1]])
COV = np.dot(P, P.T)
Y = rs.randn(d, n)
X = np.dot(P, Y).T

TRUE_ENTROPY = entropy_gaussian(COV)
RNG = list(zip(*(X.min(axis=0), X.max(axis=0))))


def test_entropy_kde():
    close(entropy(8, RNG, 'kde', X), TRUE_ENTROPY, rtol=.2)


def test_entropy_knn():
    close(entropy(3, [None], 'knn', X), TRUE_ENTROPY, rtol=.2)


def test_entropy_chaowangjost():
    close(entropy(8, RNG, 'chaowangjost', X), TRUE_ENTROPY, rtol=.2)


def test_entropy_grassberger():
    close(entropy(8, RNG, 'grassberger', X), TRUE_ENTROPY, rtol=.2)


def test_entropy_doanes_rule():
    close(entropy(None, RNG, 'grassberger', X), TRUE_ENTROPY, atol=2., rtol=.2)


def test_entropy_naive():
    close(entropy(8, RNG, None, X), TRUE_ENTROPY, rtol=.2)
