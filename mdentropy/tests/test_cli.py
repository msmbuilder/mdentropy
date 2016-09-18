import subprocess
from distutils.spawn import find_executable

MDENT = find_executable('mdent')


def test_mdent():
    assert MDENT is not None
    subprocess.check_call([MDENT, '-h'])


def test_dmutinf():
    assert MDENT is not None
    subprocess.check_call([MDENT, 'dmutinf', '-h'])


def test_dtent():
    assert MDENT is not None
    subprocess.check_call([MDENT, 'dtent', '-h'])
