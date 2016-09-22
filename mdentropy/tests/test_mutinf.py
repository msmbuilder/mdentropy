from ..utils import entropy_gaussian
from ..core import mutinf, nmutinf
from ..metrics import (AlphaAngleMutualInformation, ContactMutualInformation,
                       DihedralMutualInformation)

from msmbuilder.example_datasets import FsPeptide

import numpy as np
from numpy.testing import assert_almost_equal as eq, assert_allclose as close


rs = np.random.RandomState(42)
n = 50000
P = np.array([[1, 0], [0.5, 1]])
COV = np.dot(P, P.T)
U = rs.randn(2, n)
X, Y = np.dot(P, U)
X, Y = X.reshape(n, 1), Y.reshape(n, 1)

TRUE_MUTINF = (entropy_gaussian(COV[0, 0]) + entropy_gaussian(COV[1, 1]) -
               entropy_gaussian(COV))


def test_mutinf_kde():
    close(mutinf(8, X, Y, method='kde'), TRUE_MUTINF, atol=.01, rtol=.2)


def test_mutinf_knn():
    close(mutinf(3, X, Y, method='knn'), TRUE_MUTINF, atol=.01, rtol=.2)


def test_mutinf_chaowangjost():
    close(mutinf(8, X, Y, method='chaowangjost'), TRUE_MUTINF, atol=.01,
          rtol=.2)


def test_mutinf_grassberger():
    close(mutinf(8, X, Y, method='grassberger'), TRUE_MUTINF, atol=.01,
          rtol=.2)


def test_mutinf_doanes_rule():
    close(mutinf(None, X, Y, method='grassberger'), TRUE_MUTINF, atol=.01,
          rtol=.2)


def test_mutinf_naive():
    close(mutinf(8, X, Y, method=None), TRUE_MUTINF, atol=.01, rtol=.2)


def test_mutinf_reversible():
    MI1 = mutinf(24, X, Y)
    MI2 = mutinf(24, Y, X)

    eq(MI1, MI2, 5)


def test_nmutinf_reversible():
    MI1 = nmutinf(24, X, Y)
    MI2 = nmutinf(24, Y, X)

    eq(MI1, MI2, 5)


def test_fs_mutinf():

    traj = FsPeptide().get().trajectories[0]

    idx = [at.index for at in traj.topology.atoms
           if at.residue.index in [3, 4, 5, 6, 7, 8]]
    traj = traj.atom_slice(atom_indices=idx)[::100]

    yield _test_mi_alpha, traj
    yield _test_mi_contact, traj
    yield _test_mi_dihedral, traj


def _test_mi_alpha(traj):
    mi = AlphaAngleMutualInformation()
    M = mi.partial_transform(traj)

    eq(M - M.T, 0)


def _test_mi_contact(traj):
    mi = ContactMutualInformation()
    M = mi.partial_transform(traj)

    eq(M - M.T, 0)


def _test_mi_dihedral(traj):
    mi = DihedralMutualInformation()
    M = mi.partial_transform(traj)

    eq(M - M.T, 0)
    _test_mi_shuffle(mi, traj)


def _test_mi_shuffle(mi, traj):
    M = mi.partial_transform(traj, shuffle=0)
    MS = mi.partial_transform(traj, shuffle=1)

    error = np.abs(M - MS).ravel()

    assert any(error > 1E-6)
