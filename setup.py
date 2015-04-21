"""MDEntropy: Analyze correlated motions in MD trajectories with only a few lines of Python code.

MDEntropy is a python library that allows users to perform information-theoretic
analyses on molecular dynamics (MD) trajectories.
"""

from setuptools import setup, find_packages


classifiers = """\
    Development Status :: 3 - Alpha
    Intended Audience :: Science/Research
    License :: OSI Approved :: Apache Software License
    Programming Language :: Python
    Programming Language :: Python :: 2.6
    Programming Language :: Python :: 2.7
    Operating System :: Unix
    Operating System :: MacOS
    Operating System :: Microsoft :: Windows
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Information Analysis"""

setup(
    name = "MDEntropy",
    version = "0.1",
    packages = find_packages(),
    scripts = ['dmutinf', 'dtent.py'],
    zip_safe=True,
    platforms=["Windows", "Linux", "Mac OS-X", "Unix"],
    classifiers=[e.strip() for e in classifiers.splitlines()],
    author = "Carlos Xavier Hernández",
    author_email = "me@example.com",
    description = "This is an Example Package",
    license = "MIT",
    keywords = "molecular dynamics entropy analysis",
    url = "http://github.com/cxhernandez/mdentropy"
)
