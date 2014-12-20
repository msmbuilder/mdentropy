import numpy as np
import mdtraj as md
import argparse, cPickle
from multiprocessing import Pool
from itertools import product
from itertools import combinations_with_replacement as combinations
from contextlib import closing

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
    
def ce(X,Y):
    return ent2D(X, Y) - ent1D(Y)

def cmi(X, Y, Z):
    return ent2D(X, Z) + ent2D(Y, Z) - ent3D(X, Y, Z) - ent1D(Z)
    
def dihedrals(traj):
    kinds = [md.compute_phi, 
             md.compute_psi]
    return [kind(traj)[1].T for kind in kinds]
    
def f(cD, pD): 
    g = lambda i: sum([cmi(cD[d[0]][i[0]], pD[d[1]][i[1]], pD[d[0]][i[0]]) for d in combinations(range(len(cD)), 2)])
    g.__name__ = 'g'
    return g
    
def h(cD, pD): 
    q = lambda i: sum([ce(cD[d[0]][i[0]], pD[d[0]][i[0]]) for d in combinations(range(len(cD)), 2)])
    q.__name__ = 'q'
    return q

def run(current, past, iter, N):
    cD = dihedrals(current)
    pD = dihedrals(past)
    n = cD[0].shape[0]
    R = []
    q = h(cD, pD)
    for i in range(iter+1):
        g = f(cD, pD)
        with closing(Pool(processes=N)) as pool:
            R.append(np.reshape(pool.map(g, product(range(n), range(n))), (n, n)))
            pool.terminate()
        [np.random.shuffle(d) for d in cD]
        [np.random.shuffle(d) for d in pD]
    CMI = R[0] - np.mean(R[1:], axis = 0)
    with closing(Pool(processes=N)) as pool:
        CH = (pool.map(q, zip(*(2*[range(n)])))*np.ones((n,n))).T
        pool.terminate()
    T = CMI/CH
    return T.T - T
    
    
def parse_cmdln():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--current', dest='current',help='File containing current step states.')
    parser.add_argument('-p', '--past', dest='past',help='File containing past step states.')
    parser.add_argument('-t', '--topology', dest='top',help='File containing topology.', default=None)
    parser.add_argument('-s', '--shuffle-iter', dest='iter', help='Number of shuffle iterations.', default=100, type=int)
    parser.add_argument('-n', '--n-proc', dest='N',help='Number of processors', default=4, type=int)
    parser.add_argument('-o', '--output', dest='out', help='Name of output file.', default='tent.pkl')
    args = parser.parse_args()
    return args
    
if __name__=="__main__":
    options = parse_cmdln()
    current = md.load(options.current, top = options.top)
    past = md.load(options.past, top = options.top)
    D = run(current, past, options.iter, options.N)
    cPickle.dump(D, open(options.out, 'wb'))
