import os
import shutil
import tempfile

import numpy as np
from numpy.testing import assert_almost_equal as eq

from ..core import mi, nmi
from ..metrics import (AlphaAngleMutualInformation, ContactMutualInformation,
                       DihedralMutualInformation)

import mdtraj as md
from msmbuilder.example_datasets import FsPeptide


def test_mi():
    a = np.random.uniform(low=-180., high=180, size=1000)
    b = np.random.uniform(low=-180., high=180, size=1000)

    MI1 = mi(24, a, b, rng=[-180., 180.])
    MI2 = mi(24, b, a, rng=[-180., 180.])

    eq(MI1, MI2, 6)


def test_nmi():
    a = np.random.uniform(low=-180., high=180, size=1000)
    b = np.random.uniform(low=-180., high=180, size=1000)

    MI1 = nmi(24, a, b, rng=[-180., 180.])
    MI2 = nmi(24, b, a, rng=[-180., 180.])

    eq(MI1, MI2, 6)


def test_fs_mi():

    cwd = os.path.abspath(os.curdir)
    dirname = tempfile.mkdtemp()
    FsPeptide(dirname).get()

    try:
        os.chdir(dirname)

        top = md.load(dirname + '/fs_peptide/fs-peptide.pdb')
        idx = [at.index for at in top.topology.atoms
               if at.residue.index in [3, 4, 5, 6, 7, 8]]
        traj = md.load(dirname + '/fs_peptide/trajectory-1.xtc', stride=100,
                       top=top, atom_indices=idx)

        yield _test_mi_alpha, traj
        yield _test_mi_contact, traj
        yield _test_mi_dihedral, traj

    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)


def _test_mi_alpha(traj):
    mi = AlphaAngleMutualInformation()
    M = mi.partial_transform(traj)

    uidx = np.triu_indices(mi.labels.size)
    lidx = np.tril_indices(mi.labels.size)

    eq(M[uidx], M[lidx])


def _test_mi_contact(traj):
    mi = ContactMutualInformation()
    M = mi.partial_transform(traj)

    uidx = np.triu_indices(mi.labels.size)
    lidx = np.tril_indices(mi.labels.size)

    eq(M[uidx], M[lidx])


def _test_mi_dihedral(traj):
    mi = DihedralMutualInformation()
    M = mi.partial_transform(traj)

    uidx = np.triu_indices(mi.labels.size)
    lidx = np.tril_indices(mi.labels.size)

    eq(M[uidx], M[lidx])
    _test_mi_shuffle(mi, traj)


def _test_mi_shuffle(mi, traj):
    M = mi.partial_transform(traj, shuffle=0)
    MS = mi.partial_transform(traj, shuffle=1)

    error = np.abs(M - MS).ravel()

    assert any(error > 1E-6)
