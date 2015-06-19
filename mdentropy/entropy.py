import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score


def ent(nbins, range, correction=True, *args):
    data = np.vstack((args)).T
    hist = np.histogramdd(data, bins=nbins, range=range)
    #dx = hist[1][0][1] - hist[1][0][0]
    H = stats.entropy(hist[0].flatten())
    if correction:
        return H + panzeritreves(hist[0].flatten(), data.shape[0])
    return H


def mi(nbins, X, Y, range=2*[[-180., 180.]]):
    H = np.histogram2d(X, Y, bins=nbins, range=range)
    return mutual_info_score(None, None, contingency=H[0])


def ce(nbins, X, Y, range=[-180., 180.]):
    return ent(nbins, 2*[range], X, Y) - ent(nbins, [range], Y)


def cmi(nbins, X, Y, Z, range=[-180., 180.]):
    return sum([ent(nbins, 2*[range], X, Z),
                ent(nbins, 2*[range], Y, Z),
                -ent(nbins, 3*[range], X, Y, Z),
                - ent(nbins, [range], Z)])


def panzeritreves(binprob, totaltrials):
    totalbins = len(binprob)
    eps = np.spacing(1)
    nb = sum(binprob > eps)
    if (nb < totalbins):
        nb_x = nb - sum((binprob > eps) * (binprob < 1) * np.exp(np.log(1 - binprob + eps) * totaltrials))
        delta_N_prev = totalbins
        delta_N = abs(nb - nb_x)
        xtr = 0
        while (delta_N < delta_N_prev and ((nb+xtr) < totalbins)):
            xtr += 1
            nb_x = 0.0
            gg = xtr*(1.-((totaltrials/(totaltrials+nb))**(1./totaltrials)))
            qc_x = (1-gg) * (binprob*totaltrials+1) / (totaltrials+nb)
            nb_x = np.sum((binprob > eps) * (1 - np.exp(np.log(1 - qc_x) * totaltrials)))
            qc_x = gg / xtr
            nb_x = nb_x + xtr*(1. - np.exp(np.log(1. - qc_x) * totaltrials))
            delta_N_prev = delta_N
            delta_N = abs(nb - nb_x)
        nb += xtr - 1
        if (delta_N < delta_N_prev):
            nb += 1
    return nb
