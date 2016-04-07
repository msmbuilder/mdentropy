import subprocess
from distutils.spawn import find_executable

DMUTINF = find_executable('dmutinf')
DTENT = find_executable('dtent')


def test_dmutinf():
    subprocess.check_call([DMUTINF, '-h'])


def test_dtent():
    subprocess.check_call([DTENT, '-h'])
