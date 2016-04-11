import os
import shutil
import tempfile

import numpy as np
from numpy.testing import assert_almost_equal as eq

from ..metrics import DihedralTransferEntropy
from ..core import cmi, ce, ncmi

import mdtraj as md
from msmbuilder.example_datasets import FsPeptide


def test_ncmi():
    a = np.random.uniform(low=-180., high=180, size=1000)
    b = np.random.uniform(low=-180., high=180, size=1000)
    c = np.random.uniform(low=-180., high=180, size=1000)

    NCMI_REF = (cmi(30, a, b, c, rng=[-180., 180.]) /
                ce(30, a, c, rng=[-180., 180.]))
    NCMI = ncmi(30, a, b, c, rng=[-180., 180.])

    eq(NCMI, NCMI_REF, 6)


def test_dihedral_tent():

    cwd = os.path.abspath(os.curdir)
    dirname = tempfile.mkdtemp()
    FsPeptide(dirname).get()

    try:
        os.chdir(dirname)

        top = md.load(dirname + '/fs_peptide/fs-peptide.pdb')
        idx = [at.index for at in top.topology.atoms
               if at.residue.index in [4, 5, 6]]
        traj1 = md.load(dirname + '/fs_peptide/trajectory-1.xtc', stride=10,
                        top=top, atom_indices=idx)
        traj2 = md.load(dirname + '/fs_peptide/trajectory-2.xtc', stride=10,
                        top=top, atom_indices=idx)
        traj = (traj1, traj2)

        tent = DihedralTransferEntropy(method='symbolic')
        T = tent.partial_transform(traj)

        if T[0, 1] == T[1, 0]:
            raise ValueError('Transfer entropy test failed')

        _test_shuffle(tent, traj)

    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)


def _test_shuffle(tent, traj):
    tent.partial_transform(traj, shuffled=True)
