import numpy as np
from mdentropy.entropy import cmi, ce, ncmi


def check_ncmi():
    A = np.random.uniform(low=-180., high=180, size=1000)
    B = np.random.uniform(low=-180., high=180, size=1000)
    C = np.random.uniform(low=-180., high=180, size=1000)

    NCMI = ncmi(30, A, B, C)

    if not (NCMI == cmi(30, A, B, C)/ce(30, A, B, C)):
        ValueError('Normalized mutual information test failed.')
