from __future__ import division
import numpy as np
from scipy import stats


def bootstrap(a, n_boot=10000, stat_func=np.mean):
    """Resample an array with replacement and calculate a summary stat.

    Parameters
    ----------
    a: array
        data to resample
    n_boot: int
        number of resamples
    stat_func: callable
        function to call on each resampled dataset

    """
    boot_dist = np.zeros(n_boot)
    n = len(a)
    for i in xrange(n_boot):
        sample = a[np.random.randint(0, n, n)]
        boot_dist[i] = stat_func(sample)
    return boot_dist


def percentiles(a, pcts):
    """Like scoreatpercentile but can take and return array of percentiles.

    Parameters
    ----------
    a: array
        data
    pcts: sequence of percentile values
        percentiles to find score at

    Returns
    -------
    scores: array
        array of scores at requested percentiles
    """
    try:
        scores = np.zeros(len(pcts))
    except TypeError:
        pcts = [pcts]
        scores = np.zeros(1)
    for i, p in enumerate(pcts):
        scores[i] = stats.scoreatpercentile(a, p)
    return scores


def pmf_hist(a, bins=10):
    """Return arguments to plt.bar for pmf-like histogram of an array.

    Parameters
    ----------
    a: array-like
        array to make histogram of
    bins: int
        number of bins

    Returns
    -------
    x: array
        left x position of bars
    h: array
        height of bars
    w: float
        width of bars

    """
    n, x = np.histogram(a, bins)
    h = n / n.sum()
    w = x[1] - x[0]
    return x[:-1], h, w
