import numpy as np
from scipy import stats
import pandas as pd
from IPython import embed


def gen_cov(n):
    A = np.random.rand(n, n)
    return np.dot(A, A.T)


def shuffle(df, n=1):
    ind = df.index
    sampler = np.random.permutation
    for i in range(n):
        new_vals = df.take(sampler(df.shape[0])).values
        df = pd.DataFrame(new_vals, index=ind)
    return df


class Theoretical(object):

    def _ent(self, *args):
        f = stats.multivariate_normal(
            mean=np.array([self.U[i] for i in args]),
            cov=np.array([[self.S[i, j] for j in args] for i in args]))
        return f.entropy()

    def _ce(self, X, Y):
        return self._ent(X, Y) - self._ent(Y)

    def _cmi(self, X, Y, Z):
        return np.sum([self._ent(X, Z),
                       self._ent(Y, Z),
                       -self._ent(X, Y, Z),
                       -self._ent(Z)])

    def tent(self):
        for i in range(self.N):
            for j in range(self.N):
                self.A[i, j] = self._cmi(i, j+self.N, i+self.N)

        for i in range(self.N):
            self.B[i] = self._ce(i, i+self.N)

        T = self.A/(self.B*np.ones((self.N, self.N))).T
        T -= T.T
        return T

    def __init__(self, U, S, N):
        self.N = N
        self.U = U
        self.S = S
        self.A = np.zeros((self.N, self.N))
        self.B = np.zeros((self.N,))


class Empirical(object):

    def _ent(self, *args):
        W = np.vstack((args)).T
        n = len(args)
        H = np.histogramdd(W, bins=self.nbins, range=n*[[-180, 180]])
        dx = H[1][0][1] - H[1][0][0]
        return stats.entropy(H[0].flatten()) + n*np.log(dx)

    def _ce(self, X, Y):
        return self._ent(X, Y) - self._ent(Y)

    def _cmi(self, X, Y, Z):
        return np.sum([self._ent(X, Z),
                       self._ent(Y, Z),
                       -self._ent(X, Y, Z),
                       -self._ent(Z)])

    def tent(self):
        self.B = np.zeros((self.N,))
        self.R = []
        current = self.current.copy()
        past = self.past.copy()
        for i in range(11):
            self.A = np.zeros((self.N, self.N))
            for i in range(self.N):
                for j in range(self.N):
                    self.A[i, j] = self._cmi(current[i],
                                             past[j],
                                             past[i])
            current = shuffle(current)
            past = shuffle(past)
            self.R.append(self.A)
        self.A = self.R[0] - np.mean(self.R[1:], axis=0)
        for i in range(self.N):
            self.B[i] = self._ce(self.current[i], self.past[i])
        T = self.A/(self.B*np.ones((self.N, self.N))).T
        T -= T.T
        return T

    def __init__(self, U, S, N, n):
        self.N = N
        self.nbins = 30
        self.current = pd.DataFrame(stats.multivariate_normal(
            mean=U[:N], cov=S[:N, :N]).rvs(size=n), columns=range(N))
        self.past = pd.DataFrame(stats.multivariate_normal(
            mean=U[N:2*N], cov=S[N:2*N, N:2*N]).rvs(size=n), columns=range(N))


def run(N=100, n=1000):
    U = 180*np.random.randn(2*N)
    S = gen_cov(2*N)
    T = Theoretical(U, S, N).tent()
    E = Empirical(U, S, N, n).tent()
    print(np.sum((E-T)**2))
    embed()

if __name__ == "__main__":
    run()
