from mdentropy.entropy import nmi
from itertools import combinations_with_replacement as combinations


class MutualInformation(object):
    def __call__(self, i):
        return sum([nmi(self.n, d[0][i[0]], d[1][i[1]], method='chaowangjost')
                    if i[0] in d[0].columns and
                    i[1] in d[1].columns
                    else 0.0
                    for d in combinations(self.D, 2)])

    def __init__(self, nbins, D, method='chaowangjost'):
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
