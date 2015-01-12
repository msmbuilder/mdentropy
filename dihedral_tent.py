import numpy as np
import mdtraj as md
import time
import argparse
import cPickle
from multiprocessing import Pool
from itertools import product
from itertools import combinations_with_replacement as combinations
from contextlib import closing


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


def rbins(n=30):
    return np.linspace(-np.pi, np.pi, n+3)[1:-1]


def ent(H):
    H /= H.sum()
    return -np.sum(H*np.nan_to_num(np.log2(H)))


def ent1D(X, r=rbins()):
    H = np.histogram(X, r)[0]
    return ent(H)


def ent2D(X, Y, r=rbins()):
    H = np.histogram2d(X, Y, 2*[r])[0]
    return ent(H)


def ent3D(X, Y, Z, r=rbins()):
    W = np.vstack((X, Y, Z)).T
    H = np.histogramdd(W, 3*[r])[0]
    return ent(H)


def ce(X, Y):
    return ent2D(X, Y) - ent1D(Y)


def cmi(X, Y, Z):
    return ent2D(X, Z) + ent2D(Y, Z) - ent3D(X, Y, Z) - ent1D(Z)


def dihedrals(traj):
    kinds = [md.compute_phi,
             md.compute_psi]
    return [kind(traj)[1].T for kind in kinds]


class f(object):
    def __call__(self, i):
        return sum([cmi(self.cD[d[0]][i[0]], self.pD[d[1]][i[1]],
                    self.pD[d[0]][i[0]])
                    for d in combinations(range(len(self.cD)), 2)])

    def __init__(self, cD, pD):
        self.cD = cD
        self.pD = pD


class h(object):
    def __call__(self, i):
        return sum([ce(self.cD[d[0]][i[0]], self.pD[d[0]][i[0]])
                    for d in combinations(range(len(self.cD)), 2)])

    def __init__(self, cD, pD):
        self.cD = cD
        self.pD = pD


def run(current, past, iter, N):
    cD = dihedrals(current)
    pD = dihedrals(past)
    n = cD[0].shape[0]
    R = []
    q = h(cD, pD)
    for i in range(iter+1):
        g = f(cD, pD)
        with timing(i):
            with closing(Pool(processes=N)) as pool:
                R.append(np.reshape(pool.map(g, product(range(n),
                                             range(n))),
                                            (n, n)))
                pool.terminate()
            [np.random.shuffle(d) for d in cD]
            [np.random.shuffle(d) for d in pD]
    CMI = R[0] - np.mean(R[1:], axis=0)
    with closing(Pool(processes=N)) as pool:
        CH = (pool.map(q, zip(*(2*[range(n)])))*np.ones((n, n))).T
        pool.terminate()
    T = CMI/CH
    return T - T.T


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
    D = run(current, past, options.iter, options.N)
    cPickle.dump(D, open(options.out, 'wb'))
