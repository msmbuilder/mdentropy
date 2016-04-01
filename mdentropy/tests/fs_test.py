import os
import shutil
import tempfile

from mdentropy.mutinf import DihedralMutualInformation
from mdentropy.tent import DihedralTransferEntropy

from numpy.testing import assert_almost_equal as eq

import mdtraj as md

from msmbuilder.example_datasets import FsPeptide


def test_fs_mi():

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

    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)


def test_fs_tent():

    cwd = os.path.abspath(os.curdir)
    dirname = tempfile.mkdtemp()
    FsPeptide(dirname).get()

    try:
        os.chdir(dirname)

        top = md.load(dirname + '/fs_peptide/fs-peptide.pdb')
        traj1 = md.load(dirname + '/fs_peptide/trajectory-1.xtc', stride=10,
                        top=top)
        traj2 = md.load(dirname + '/fs_peptide/trajectory-2.xtc', stride=10,
                        top=top)
        traj = (traj1, traj2)

        tent = DihedralTransferEntropy()
        T = tent.partial_transform(traj)

        if T[0, 1] == T[1, 0]:
            raise ValueError('Transfer entropy test failed')

    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)
