import os
import shutil
import tempfile

import numpy as np
from numpy.testing import assert_almost_equal as eq

from ..metrics import DihedralMutualInformation
from ..core import mi, nmi

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


def test_dihedral_mi():

    cwd = os.path.abspath(os.curdir)
    dirname = tempfile.mkdtemp()
    FsPeptide(dirname).get()

    try:
        os.chdir(dirname)

        top = md.load(dirname + '/fs_peptide/fs-peptide.pdb')
        traj = md.load(dirname + '/fs_peptide/trajectory-1.xtc',
                       stride=10, top=top)

        mi = DihedralMutualInformation()
        M = mi.partial_transform(traj)

        eq(M[0, 1], M[1, 0])

        _test_shuffle(mi, traj)

    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)


def _test_shuffle(mi, traj):
    mi.partial_transform(traj, shuffled=True)
