import numpy as np
from scipy import stats


def bootstrap(a, n_boot=10000, stat_func=np.mean):

    boot_dist = np.zeros(n_boot)
    n = len(a)
    for i in xrange(n_boot):
        sample = a[np.random.randint(0, n, n)]
        boot_dist[i] = stat_func(sample)
    return boot_dist


def percentiles(a, pcts):

    out = np.zeros(len(pcts))
    for i, p in enumerate(pcts):
        out[i] = stats.scoreatpercentile(a, p)
    return out
