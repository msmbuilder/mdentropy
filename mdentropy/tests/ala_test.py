import os
import shutil
import tempfile

from mdentropy.utils import dihedrals
from mdentropy.mutinf import MutualInformation

from msmbuilder.example_datasets import AlanineDipeptide
import mdtraj as md


def test_ala_mi():
    cwd = os.path.abspath(os.curdir)
    dirname = tempfile.mkdtemp()
    AlanineDipeptide(dirname).get()

    try:
        os.chdir(dirname)

        top = md.load(dirname + 'ala2.pdb')
        traj = md.load(dirname + 'trajectory-0.dcd', top=top)
        d = dihedrals(traj)

        mi = MutualInformation(d)

        if mi((0, 1)) != mi((1, 0)):
            raise ValueError("Mutual information is not symmetric.")

    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)
