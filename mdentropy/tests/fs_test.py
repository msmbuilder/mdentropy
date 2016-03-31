import os
import shutil
import tempfile

from mdentropy.mutinf import DihedralMutualInformation
from mdentropy.tent import DihedralTransferEntropy

from numpy.testing import assert_almost_equal as eq

import mdtraj as md

from msmbuilder.example_datasets import FsPeptide

CWD = os.path.abspath(os.curdir)
DIRNAME = tempfile.mkdtemp()
FsPeptide(DIRNAME).get()


def test_fs_mi():

    try:
        os.chdir(DIRNAME)

        top = md.load(DIRNAME + 'fs_peptide/fs-peptide.pdb')
        traj = md.load(DIRNAME + 'fs_peptide/trajectory-1.xtc', top=top)

        mi = DihedralMutualInformation()
        M = mi.partial_transform(traj)

        eq(M[0, 1], M[1, 0])

    finally:
        os.chdir(CWD)


def test_fs_tent():
    try:
        os.chdir(DIRNAME)

        top = md.load(DIRNAME + '/fs_peptide/fs-peptide.pdb')
        traj = md.load(DIRNAME + '/fs_peptide/trajectory-1.xtc', top=top)

        tent = DihedralTransferEntropy()
        T = tent.partial_transform(traj)

        if T[0, 1] == T[1, 0]:
            raise ValueError('Transfer entropy test failed')

    finally:
        os.chdir(CWD)
        shutil.rmtree(DIRNAME)
