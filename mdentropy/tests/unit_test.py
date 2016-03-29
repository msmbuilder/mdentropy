import numpy as np
from mdentropy.entropy import cmi, ce, ncmi, nmi


def test_ncmi():
    A = np.random.uniform(low=-180., high=180, size=1000)
    B = np.random.uniform(low=-180., high=180, size=1000)
    C = np.random.uniform(low=-180., high=180, size=1000)

    NCMI = ncmi(30, A, B, C)

    if not (NCMI == cmi(30, A, B, C)/ce(30, A, B, C)):
        raise ValueError('Normalized conditional mutual '
                         'information test failed.')


def test_mi():
    A = np.random.uniform(low=-180., high=180, size=1000)
    B = np.random.uniform(low=-180., high=180, size=1000)

    MI1 = nmi(24, A, B)
    MI2 = nmi(24, B, A)

    if not MI1 == MI2:
        raise ValueError('Normalized mutual information test failed.')
