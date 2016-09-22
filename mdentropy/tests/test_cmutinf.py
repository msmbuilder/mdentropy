
from ..utils import entropy_gaussian
from ..core import cmutinf, centropy, ncmutinf
from ..metrics import (AlphaAngleTransferEntropy, ContactTransferEntropy,
                       DihedralTransferEntropy)

from msmbuilder.example_datasets import FsPeptide

import numpy as np
from numpy.testing import assert_almost_equal as eq, assert_allclose as close

rs = np.random.RandomState(42)
n, d = 50000, 3

P = np.array([[1, .5, .25], [.5, 1, 0], [.25, 0, 1]])
COV = np.dot(P, P.T)
Y = rs.randn(d, n)
a, b, c = np.dot(P, Y)
a, b, c = np.atleast_2d(a).T, np.atleast_2d(b).T, np.atleast_2d(c).T

true_cmutinf = (entropy_gaussian(COV[[[0, 0], [0, 2]], [[0, 2], [2, 2]]]) +
                entropy_gaussian(COV[[[1, 1], [1, 2]], [[1, 2], [2, 2]]]) -
                entropy_gaussian(COV) - entropy_gaussian(COV[2, 2]))
true_cond_ent = (entropy_gaussian(COV[[[0, 0], [0, 2]], [[0, 2], [2, 2]]]) -
                 entropy_gaussian(COV[2, 2]))

TRUE_NCMUTINF = true_cmutinf / true_cond_ent


def test_ncmutinf_kde():
    close(ncmutinf(3, a, b, c, method='kde'), TRUE_NCMUTINF, atol=.05, rtol=.2)


def test_ncmutinf_knn():
    close(ncmutinf(3, a, b, c, method='knn'), TRUE_NCMUTINF, atol=.05, rtol=.2)


def test_ncmutinf_chaowangjost():
    close(ncmutinf(8, a, b, c, method='chaowangjost'), TRUE_NCMUTINF, atol=.05,
          rtol=.2)


def test_ncmutinf_grassberger():
    close(ncmutinf(8, a, b, c, method='grassberger'), TRUE_NCMUTINF, atol=.05,
          rtol=.2)


def test_ncmutinf_doanes_rule():
    close(ncmutinf(None, a, b, c, method='grassberger'), TRUE_NCMUTINF,
          atol=.05, rtol=.4)


def test_ncmutinf_naive():
    close(ncmutinf(8, a, b, c, method=None), TRUE_NCMUTINF, atol=.05, rtol=.2)


def test_ncmutinf():
    a = rs.uniform(low=0, high=360, size=1000).reshape(-1, 1)
    b = rs.uniform(low=0, high=360, size=1000).reshape(-1, 1)
    c = rs.uniform(low=0, high=360, size=1000).reshape(-1, 1)

    NCMI_REF = (cmutinf(10, a, b, c) /
                centropy(10, a, c))
    NCMI = ncmutinf(10, a, b, c)

    eq(NCMI, NCMI_REF, 5)


def test_fs_tent():

    traj1, traj2 = FsPeptide().get().trajectories[:2]

    idx = [at.index for at in traj1.topology.atoms
           if at.residue.index in [3, 4, 5, 6, 7, 8]]

    traj1 = traj1.atom_slice(atom_indices=idx)[::100]
    traj2 = traj2.atom_slice(atom_indices=idx)[::100]

    traj = (traj1, traj2)

    yield _test_tent_alpha, traj
    yield _test_tent_contact, traj
    yield _test_tent_dihedral, traj


def _test_tent_alpha(traj):
    tent = AlphaAngleTransferEntropy()
    T = tent.partial_transform(traj)

    assert T is not None


def _test_tent_contact(traj):
    tent = ContactTransferEntropy()
    T = tent.partial_transform(traj)

    assert T is not None


def _test_tent_dihedral(traj):
    tent = DihedralTransferEntropy()
    T = tent.partial_transform(traj)

    assert T is not None
    _test_tent_shuffle(tent, traj)


def _test_tent_shuffle(tent, traj):
    T = tent.partial_transform(traj, shuffle=0)
    TS = tent.partial_transform(traj, shuffle=1)

    assert T is not None
    assert TS is not None
