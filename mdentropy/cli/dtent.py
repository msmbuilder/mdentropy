# !/usr/bin/env python

from argparse import ArgumentDefaultsHelpFormatter


def func(args, parser):
    # delay import of the rest of the module to improve `mdentropy -h`
    # performance
    import pickle
    import mdtraj as md
    import pandas as pd
    from ..utils import parse_files
    from ..metrics import DihedralTransferEntropy

    f1, f2 = parse_files(args.current), parse_files(args.past)

    current = md.load(f1, top=args.top, stride=args.stride)
    past = md.load(f2, top=args.top, stride=args.stride)

    tent = DihedralTransferEntropy(n_bins=args.nbins, types=args.types,
                                   method=args.method, threads=args.n_threads,
                                   normed=True)

    T = tent.partial_transform((past, current), shuffle=iter, verbose=True)

    df = pd.DataFrame(T, columns=tent.labels)

    pickle.dump(df, open(args.out, 'wb'))


def configure_parser(sub_parsers):
    help = 'Run a dihedral transfer entropy calculation'
    p = sub_parsers.add_parser('dtent', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('-p', '--past', dest='past',
                   help='File containing past step states.',
                   required=True)
    p.add_argument('-s', '--shuffle-iter', dest='iter',
                   help='Number of shuffle iterations.',
                   default=10, type=int)
    p.add_argument('-r', '--stride', dest='stride',
                   help='Stride to use', default=1, type=int)
    p.add_argument('-t', '--topology',
                   dest='top', help='File containing topology.',
                   default=None)
    p.add_argument('-b', '--n-bins', dest='nbins',
                   help='Number of bins', default=3, type=int)
    p.add_argument('-n', '--n-threads', dest='N',
                   help='Number of threads', default=None, type=int)
    p.add_argument('-o', '--output', dest='out',
                   help='Name of output file.', default='tent.pkl')
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
