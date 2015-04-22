from mdentropy.entropy import mi


class MutualInformationFactory(object):
    def fit(cls, *args):
        cls._fit(*args)

    def transform(cls):
        return cls._mutinf


class MutualInformation(MutualInformationFactory):

    def _fit(self, X, Y):
        if self.range is None:
            self.range = 2*[[min(min(X), min(Y)), max(max(X), max(Y))]]
        return mi(self.nbins, X, Y, range=self.range)

    def __init__(self, shuffle=0, nbins=24, range=None):
        return None


class DihedralMutualInformation(MutualInformation):

    def _fit(self, X, Y):
        return mi()


class CartesianMutualInformation(MutualInformation):

    def _fit(self, X, Y):
        return mi()
