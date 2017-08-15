import numpy as np
from joblib import Parallel, delayed, cpu_count


def parallel_pairwise(x, Y, n_jobs, func, func_params, **kwds):
    """Break the pairwise matrix in n_jobs even slices
       and compute them in parallel"""

    if n_jobs < 0:
        n_jobs = max(cpu_count() + 1 + n_jobs, 1)

    if n_jobs == 1:
        return func_batch(x, Y, func, **func_params)
    else:
        # TODO: in some cases, backend='threading' may be appropriate
        fd = delayed(func_batch)
        ret = Parallel(n_jobs=n_jobs, verbose=0)(
            fd(x, Y[s], func, func_params)
            for s in gen_even_slices(len(Y), n_jobs)
        )
    return np.hstack(ret)


def func_batch(x, Y, func, func_params, **kwds):
    dist = []
    for i in range(len(Y)):
        val = func(x, Y[i], **func_params)
        dist.append(val)
    return np.array(dist)


def gen_even_slices(n, n_packs, n_samples=None):
    """Generator to create n_packs slices going up to n.

        Pass n_samples when the slices are to be used for sparse matrix indexing;
        slicing off-the-end raises an exception, while it works for NumPy arrays.

        Examples
        --------
        >>> from sklearn.utils import gen_even_slices
        >>> list(gen_even_slices(10, 1))
        [slice(0, 10, None)]
        >>> list(gen_even_slices(10, 10))                     #doctest: +ELLIPSIS
        [slice(0, 1, None), slice(1, 2, None), ..., slice(9, 10, None)]
        >>> list(gen_even_slices(10, 5))                      #doctest: +ELLIPSIS
        [slice(0, 2, None), slice(2, 4, None), ..., slice(8, 10, None)]
        >>> list(gen_even_slices(10, 3))
        [slice(0, 4, None), slice(4, 7, None), slice(7, 10, None)]
    """
    start = 0
    if n_packs < 1:
        raise ValueError("gen_even_slices got n_packs=%s, must be >=1" % n_packs)
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            if n_samples is not None:
                end = min(n_samples, end)
            yield slice(start, end, None)
            start = end
