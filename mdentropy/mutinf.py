from mdentropy.entropy import nmi
from itertools import product


class MutualInformation(object):
    def __call__(self, i):
        return sum([0.0 if (i[0] not in d[0].columns or
                            i[1] not in d[1].columns)
                    else 1.0
                    if i[0] == i[1] and all(d[0][i[0]].isin(d[1][i[1]]))
                    else nmi(self.n, d[0][i[0]], d[1][i[1]],
                             method=self.method)
                    for d in product(self.D, self.D)])

    def __init__(self, D, nbins=24, method='chaowangjost'):
        self.D = D
        self.n = nbins
        self.method = method

# class MutualInformationFactory(object):
#     def fit(cls, *args):
#         cls._fit(*args)

#     def transform(cls):
#         return cls._mutinf


# class MutualInformation(MutualInformationFactory):
#
#     def _fit(self, X, Y):
#         if self.range is None:
#             self.range = 2*[[min(min(X), min(Y)), max(max(X), max(Y))]]
#         return mi(self.nbins, X, Y, range=self.range)
#
#     def __init__(self, shuffle=0, nbins=24, range=None):
#         return None


# class DihedralMutualInformation(MutualInformation):
#
#     def _fit(self, X, Y):
#         return mi()


# class CartesianMutualInformation(MutualInformation):
#
#     def _fit(self, X, Y):
#         return mi()
