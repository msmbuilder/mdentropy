import numpy as np
import mdtraj as md
import argparse
import cPickle
from multiprocessing import Pool
from itertools import combinations_with_replacement as combinations
from sklearn.metrics import mutual_info_score
from contextlib import closing


def rbins(n=30):
    return np.linspace(-np.pi, np.pi, n+3)[1:-1]


def mi(X, Y, r=rbins()):
    H = np.histogram2d(X, Y, [r, r])[0]
    return mutual_info_score(None, None, contingency=H)


def dihedrals(traj):
    kinds = [md.compute_phi,
             md.compute_psi]
    return [kind(traj)[1].T for kind in kinds]


def f(D):
    def g(i):
        sum([mi(d[0][i[0]], d[1][i[1]]) for d in combinations(D, 2)])

    return g


def run(traj, iter, N):
    D = dihedrals(traj)
    n = D[0].shape[0]
    R = []
    for i in range(iter+1):
        r = np.zeros((n, n))
        g = f(D)
        with closing(Pool(processes=N)) as pool:
            r[np.triu_indices(n)] = pool.map(g, combinations(range(n), 2))
            pool.terminate()
        r[np.triu_indices(n)[::-1]] = r[np.triu_indices(n)]
        R.append(r)
        [np.random.shuffle(d) for d in D]
    return R[0] - np.mean(R[1:], axis=0)


def parse_cmdln():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', dest='traj',
                        help='File containing trajectory.')
    parser.add_argument('-s', '--shuffle-iter', dest='iter',
                        help='Number of shuffle iterations.',
                        default=100, type=int)
    parser.add_argument('-t', '--topology', dest='top',
                        help='File containing topology.', default=None)
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
    M = run(traj, options.iter, options.N)
    cPickle.dump(M, open(options.out, 'wb'))
