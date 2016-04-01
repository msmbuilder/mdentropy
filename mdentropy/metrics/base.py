from mdentropy.utils import shuffle

from multiprocessing import cpu_count


class MetricBase(object):

    def _shuffle(cls):
        cls.data = shuffle(cls.data)

    def transform(cls, trajs):
        for traj in trajs:
            yield cls.partial_transform(traj)

    def __init__(cls, n_bins=24, rng=None, method='chaowangjost',
                 threads=None):
        cls.n_types = 1
        cls.data = None
        cls.labels = None
        cls.n_bins = n_bins
        cls.rng = rng
        cls.method = method
        cls.n_threads = threads or int(cpu_count()/2)
