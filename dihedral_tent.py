import time
import cPickle
import argparse
import numpy as np
import mdtraj as md
import pandas as pd
from scipy import stats
from itertools import product
from contextlib import closing
from multiprocessing import Pool
from itertools import combinations_with_replacement as combinations


class timing(object):
    "Context manager for printing performance"
    def __init__(self, iter):
        self.iter = iter

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, ty, val, tb):
        end = time.time()
        print("Round %s : %0.3f seconds" %
              (self.iter, end-self.start))
        return False


def shuffle(df, n=1):
    ind = df.index
    sampler = np.random.permutation
    for i in range(n):
        new_vals = df.take(sampler(df.shape[0])).values
        df = pd.DataFrame(new_vals, index=ind)
    return df


def ent(nbins, *args):
    W = np.vstack((args)).T
    n = len(args)
    H = np.histogramdd(W, bins=nbins, range=n*[[-180, 180]])
    dx = H[1][0][1] - H[1][0][0]
    return stats.entropy(H[0].flatten()) + n*np.log(dx)


def ce(nbins, X, Y):
    return ent(nbins, X, Y) - ent(nbins, Y)


def cmi(nbins, X, Y, Z):
    return sum([ent(nbins, X, Z),
                ent(nbins, Y, Z),
                -ent(nbins, X, Y, Z),
                - ent(nbins, Z)])


class getDihedrals(object):
    def __call__(self, traj):
        atoms, angles = self.method(traj)
        idx = [traj.topology.atom(i).residue.index
               for i in atoms[:, self.type]]
        return pd.DataFrame(180*angles/np.pi, columns=idx)

    def __init__(self, method, type):
        assert type < 3 or type > -1
        self.type = type
        self.method = method


def dihedrals(traj):
    kinds = [
        getDihedrals(md.compute_phi, 2),
        getDihedrals(md.compute_psi, 1),
        # getDihedrals(md.compute_chi1, 1),
        # getDihedrals(md.compute_chi2, 0)
        ]
    return [kind(traj) for kind in kinds]


class F(object):
    def __call__(self, i):
        return sum([cmi(self.n, self.cD[d[0]][i[0]], self.pD[d[1]][i[1]],
                        self.pD[d[0]][i[0]])
                    if i[0] in self.cD[d[0]].columns
                    and i[1] in self.cD[d[1]].columns
                    else 0.0
                    for d in combinations(range(len(self.cD)), 2)])

    def __init__(self, nbins, cD, pD):
        self.cD = cD
        self.pD = pD
        self.n = nbins


class H(object):
    def __call__(self, i):
        return sum([ce(self.n, self.cD[d[0]][i], self.pD[d[1]][i])
                    if i in self.cD[d[0]].columns
                    and i in self.pD[d[1]].columns
                    else 0.0
                    for d in combinations(range(len(self.cD)), 2)])

    def __init__(self, nbins, cD, pD):
        self.cD = cD
        self.pD = pD
        self.n = nbins


def run(current, past, nbins, iter, N):
    cD = dihedrals(current)
    pD = dihedrals(past)
    n = np.unique(np.hstack(tuple(map(np.array, [df.columns for df in cD]))))
    R = []
    q = H(nbins, cD, pD)
    for i in range(iter+1):
        g = F(nbins, cD, pD)
        with timing(i):
            with closing(Pool(processes=N)) as pool:
                R.append(np.reshape(pool.map(g, product(n, n)),
                                            (n.size, n.size)))
                pool.terminate()
            cD = [shuffle(df) for df in cD]
            pD = [shuffle(df) for df in pD]
    CMI = R[0] - np.mean(R[1:], axis=0)
    with closing(Pool(processes=N)) as pool:
        CH = (pool.map(q, n)*np.ones((n.size, n.size))).T
        pool.terminate()
    T = CMI/CH
    return pd.DataFrame(T - T.T, columns=n)


def parse_cmdln():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--current', dest='current',
                        help='File containing current step states.')
    parser.add_argument('-p', '--past', dest='past',
                        help='File containing past step states.')
    parser.add_argument('-t', '--topology',
                        dest='top', help='File containing topology.',
                        default=None)
    parser.add_argument('-s', '--shuffle-iter', dest='iter',
                        help='Number of shuffle iterations.',
                        default=100, type=int)
    parser.add_argument('-b', '--n-bins', dest='nbins',
                        help='Number of bins', default=30, type=int)
    parser.add_argument('-n', '--n-proc', dest='N',
                        help='Number of processors', default=4, type=int)
    parser.add_argument('-o', '--output', dest='out',
                        help='Name of output file.', default='tent.pkl')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    options = parse_cmdln()
    current = md.load(options.current, top=options.top)
    past = md.load(options.past, top=options.top)
    D = run(current, past, options.nbins, options.iter, options.N)
    cPickle.dump(D, open(options.out, 'wb'))
