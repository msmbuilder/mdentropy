import time
import cPickle
import argparse
import numpy as np
import mdtraj as md
import pandas as pd
from scipy import stats
from contextlib import closing
from multiprocessing import Pool
from sklearn.metrics import mutual_info_score
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


def mi(nbins, X, Y):
    H = np.histogram2d(X, Y, bins=nbins, range=2*[[-180, 180]])
    return mutual_info_score(None, None, contingency=H[0])


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


class f(object):
    def __call__(self, i):
        return sum([mi(self.n, d[0][i[0]], d[1][i[1]])
                    if i[0] in d[0].columns
                    and i[1] in d[1].columns
                    else 0.0
                    for d in combinations(self.D, 2)])

    def __init__(self, nbins, D):
        self.D = D
        self.n = nbins


def run(traj, nbins, iter, N):
    D = dihedrals(traj)
    n = np.unique(np.hstack(tuple(map(np.array, [df.columns for df in D]))))
    R = []
    for i in xrange(iter+1):
        r = np.zeros((n.size, n.size))
        g = f(nbins, D)
        with timing(i):
            idx = np.triu_indices(n.size)
            with closing(Pool(processes=N)) as pool:
                r[idx] = pool.map(g, combinations(n, 2))
                pool.terminate()
            r[idx[::-1]] = r[idx]
            R.append(r)
            D = [shuffle(df) for df in D]
    if iter > 0:
        return R[0] - np.mean(R[1:], axis=0)
    return R[0]


def parse_cmdln():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', dest='traj',
                        help='File containing trajectory.', required=True)
    parser.add_argument('-s', '--shuffle-iter', dest='iter',
                        help='Number of shuffle iterations.',
                        default=100, type=int)
    parser.add_argument('-t', '--topology', dest='top',
                        help='File containing topology.', default=None)
    parser.add_argument('-b', '--n-bins', dest='nbins',
                        help='Number of bins', default=30, type=int)
    parser.add_argument('-n', '--n-proc', dest='N',
                        help='Number of processors to be used.',
                        default=4, type=int)
    parser.add_argument('-o', '--output', dest='out',
                        help='Name of output file.', default='mutinf.pkl')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    options = parse_cmdln()
    traj = md.load(options.traj, top=options.top)
    M = run(traj, options.nbins, options.iter, options.N)
    cPickle.dump(M, open(options.out, 'wb'))
