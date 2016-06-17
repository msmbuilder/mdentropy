import os
import shutil
import tempfile

import numpy as np
from numpy.testing import assert_almost_equal as eq

from ..core import cmi, ce, ncmi
from ..metrics import (AlphaAngleTransferEntropy, ContactTransferEntropy,
                       DihedralTransferEntropy)

import mdtraj as md
from msmbuilder.example_datasets import FsPeptide


def test_ncmi():
    a = np.random.uniform(low=-180., high=180, size=1000).reshape(-1, 1)
    b = np.random.uniform(low=-180., high=180, size=1000).reshape(-1, 1)
    c = np.random.uniform(low=-180., high=180, size=1000).reshape(-1, 1)

    NCMI_REF = (cmi(30, a, b, c, rng=[-180., 180.]) /
                ce(30, a, c, rng=[-180., 180.]))
    NCMI = ncmi(30, a, b, c, rng=[-180., 180.])

    eq(NCMI, NCMI_REF, 6)


def test_fs_tent():

    cwd = os.path.abspath(os.curdir)
    dirname = tempfile.mkdtemp()
    FsPeptide(dirname).get()

    try:
        os.chdir(dirname)

        top = md.load(dirname + '/fs_peptide/fs-peptide.pdb')
        idx = [at.index for at in top.topology.atoms
               if at.residue.index in [3, 4, 5, 6, 7, 8]]
        traj1 = md.load(dirname + '/fs_peptide/trajectory-1.xtc', stride=100,
                        top=top, atom_indices=idx)
        traj2 = md.load(dirname + '/fs_peptide/trajectory-2.xtc', stride=100,
                        top=top, atom_indices=idx)
        traj = (traj1, traj2)

        yield _test_tent_alpha, traj
        yield _test_tent_contact, traj
        yield _test_tent_dihedral, traj

    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)


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
