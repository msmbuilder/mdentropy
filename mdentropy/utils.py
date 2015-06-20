from __future__ import print_function

import time
import numpy as np
import pandas as pd


class timing(object):
    "Context manager for printing performance"
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, ty, val, tb):
        end = time.time()
        print("Runtime : %0.3f seconds" %
              (end-self.start))
        return False


def shuffle(df, n=1):
    ind = df.index
    sampler = np.random.permutation
    for i in range(n):
        new_vals = df.take(sampler(df.shape[0])).values
        df = pd.DataFrame(new_vals, index=ind)
    return df
