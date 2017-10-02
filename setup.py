"""
MDEntropy: Analyze correlated motions in MD trajectories with only a few
 lines of Python.

MDEntropy is a python library that allows users to perform
 information-theoretic analyses on molecular dynamics (MD) trajectories.
"""

import sys
import subprocess

from distutils.spawn import find_executable
from setuptools import setup, find_packages
from basesetup import write_version_py

NAME = "mdentropy"
VERSION = "0.4.0dev0"
ISRELEASED = False
__version__ = VERSION


def readme_to_rst():
    pandoc = find_executable('pandoc')
    if pandoc is None:
        raise RuntimeError("Turning the readme into a description requires "
                           "pandoc.")
    long_description = subprocess.check_output(
        [pandoc, 'README.md', '-t', 'rst'])
    short_description = long_description.split('\n\n')[1]
    return {
        'description': short_description,
        'long_description': long_description,
    }


def main(**kwargs):

    write_version_py(VERSION, ISRELEASED, 'mdentropy/version.py')

    setup(
        name=NAME,
        version=VERSION,
        platforms=("Windows", "Linux", "Mac OS-X", "Unix",),
        classifiers=(
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            'Operating System :: Microsoft :: Windows',
            'Topic :: Information Analysis',
        ),
        keywords="molecular dynamics entropy analysis",
        author="Carlos Xavier Hernández",
        author_email="cxh@stanford.edu",
        url='https://github.com/msmbuilder/%s' % NAME,
        download_url='https://github.com/msmbuilder/%s/tarball/master' % NAME,
        license='MIT',
        packages=find_packages(),
        include_package_data=True,
        package_data={
            '': ['README.md',
                 'requirements.txt'],
        },
        zip_safe=False,
        entry_points={
            'console_scripts': [
                'mdent = mdentropy.cli.main:main',
            ],
        },
        **kwargs
    )


if __name__ == '__main__':
    kwargs = {}
    if any(e in sys.argv for e in ('upload', 'register', 'sdist')):
        kwargs = readme_to_rst()
    main(**kwargs)
