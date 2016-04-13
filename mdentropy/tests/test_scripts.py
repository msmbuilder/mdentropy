import subprocess
from distutils.spawn import find_executable

DMUTINF = find_executable('dmutinf')
DTENT = find_executable('dtent')


def test_dmutinf():
    assert DMUTINF is not None
    subprocess.check_call([DMUTINF, '-h'])


def test_dtent():
    assert DTENT is not None
    subprocess.check_call([DTENT, '-h'])
