import numpy as np
from numpy.testing import assert_almost_equal as eq

from ..core import mutinf, nmutinf
from ..metrics import (AlphaAngleMutualInformation, ContactMutualInformation,
                       DihedralMutualInformation)

from msmbuilder.example_datasets import FsPeptide

rs = np.random.RandomState(42)


def test_mutinf():
    a = rs.uniform(low=0., high=360., size=1000).reshape(-1, 1)
    b = rs.uniform(low=0., high=360., size=1000).reshape(-1, 1)

    MI1 = mutinf(10, a, b)
    MI2 = mutinf(10, b, a)

    eq(MI1, MI2, 6)


def test_nmutinf():
    a = rs.uniform(low=0., high=360., size=1000).reshape(-1, 1)
    b = rs.uniform(low=0., high=360., size=1000).reshape(-1, 1)

    MI1 = nmutinf(24, a, b)
    MI2 = nmutinf(24, b, a)

    eq(MI1, MI2, 6)


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
