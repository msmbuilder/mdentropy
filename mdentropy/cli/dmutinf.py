# !/usr/bin/env python

from argparse import ArgumentDefaultsHelpFormatter


def func(args, parser):
    # delay import of the rest of the module to improve `mdentropy -h` performance
    import pickle
    import mdtraj as md
    import pandas as pd
    from ..utils import parse_files
    from ..metrics import DihedralMutualInformation

    files = parse_files(args.traj)
    traj = md.load(files, top=args.top, stride=args.stride)

    mi = DihedralMutualInformation(n_bins=args.nbins, types=args.types,
                                   method=args.method, threads=args.n_threads,
                                   normed=True)

    M = mi.partial_transform(traj, shuffle=iter, verbose=True)

    df = pd.DataFrame(M, columns=mi.labels)

    pickle.dump(df, open(args.out, 'wb'))


def configure_parser(sub_parsers):
    help = 'Run a dihedral mutual information calculation'
    p = sub_parsers.add_parser('dmutinf', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--input', dest='traj',
                   help='File containing trajectory.', required=True)
    p.add_argument('-s', '--shuffle-iter', dest='iter',
                   help='Number of shuffle iterations.',
                   default=100, type=int)
    p.add_argument('-t', '--topology', dest='top',
                   help='File containing topology.', default=None)
    p.add_argument('-b', '--n-bins', dest='nbins',
                   help='Number of bins', default=3, type=int)
    p.add_argument('-n', '--n-threads', dest='n_threads',
                   help='Number of threads to be used.',
                   default=None, type=int)
    p.add_argument('-r', '--stride', dest='stride',
                   help='Stride to use', default=1, type=int)
    p.add_argument('-o', '--output', dest='out',
                   help='Name of output file.', default='mutinf.pkl')
    p.add_argument('-m', '--method', dest='method',
                   help='Entropy estimate method.',
                   choices=['chaowangjost', 'grassberger', 'kde',
                            'knn', 'naive'],
                   default='knn')
    p.add_argument('-d', '--dihedrals', dest='dihedrals',
                   help='Dihedral angles to analyze.',
                   nargs='+',
                   choices=['phi', 'psi', 'omega', 'chi1',
                            'chi2', 'chi3', 'chi4'],
                   default=['phi', 'psi'])
    p.set_defaults(func=func)
