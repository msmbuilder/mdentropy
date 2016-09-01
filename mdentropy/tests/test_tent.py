import numpy as np
from numpy.testing import assert_almost_equal as eq

from ..core import cmutinf, centropy, ncmutinf
from ..metrics import (AlphaAngleTransferEntropy, ContactTransferEntropy,
                       DihedralTransferEntropy)

from msmbuilder.example_datasets import FsPeptide

rs = np.random.RandomState(42)


def test_ncmutinf():
    a = rs.uniform(low=0, high=360, size=1000).reshape(-1, 1)
    b = rs.uniform(low=0, high=360, size=1000).reshape(-1, 1)
    c = rs.uniform(low=0, high=360, size=1000).reshape(-1, 1)

    NCMI_REF = (cmutinf(10, a, b, c) /
                centropy(10, a, c))
    NCMI = ncmutinf(10, a, b, c)

    eq(NCMI, NCMI_REF, 6)


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

    error = np.abs(T - T.T).ravel()

    assert any(error > 1E-6)


def _test_tent_contact(traj):
    tent = ContactTransferEntropy()
    T = tent.partial_transform(traj)

    error = np.abs(T - T.T).ravel()

    assert any(error > 1E-6)


def _test_tent_dihedral(traj):
    tent = DihedralTransferEntropy()
    T = tent.partial_transform(traj)

    error = np.abs(T - T.T).ravel()

    assert any(error > 1E-6)
    _test_tent_shuffle(tent, traj)


def _test_tent_shuffle(tent, traj):
    T = tent.partial_transform(traj, shuffle=0)
    TS = tent.partial_transform(traj, shuffle=1)

    error = np.abs(T - TS).ravel()

    assert any(error > 1E-6)
