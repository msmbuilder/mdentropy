import numpy as np
from numpy.testing import assert_almost_equal as eq
from mdentropy.entropy import cmi, ce, ncmi, nmi


def test_ncmi():
    A = np.random.uniform(low=-180., high=180, size=1000)
    B = np.random.uniform(low=-180., high=180, size=1000)
    C = np.random.uniform(low=-180., high=180, size=1000)

    NCMI = ncmi(30, A, B, C)

    eq(NCMI, cmi(30, A, B, C)/ce(30, A, C), 6)


def test_mi():
    A = np.random.uniform(low=-180., high=180, size=1000)
    B = np.random.uniform(low=-180., high=180, size=1000)

    MI1 = nmi(24, A, B)
    MI2 = nmi(24, B, A)

    eq(MI1, MI2, 6)
