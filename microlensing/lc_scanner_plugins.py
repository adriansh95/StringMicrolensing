import numpy as np
from numba import njit

@njit
def evaluate_enough_filters(bright_mask, min_per_filter, n_filters_req):
    result = (bright_mask.sum(axis=1) > (n_filters_req - 1))
    return result

@njit
def evaluate_bright_and_baseline(
    bright_mask,
    baseline_mask
    ):
    # In case I choose not to use Numba here,
    # bright_and_baseline = (~bright_filter_mask | baseline_filter_mask).all(axis=1)
    # is equivalent but quite a bit slower.
    m = ~bright_mask
    result = np.array(
        [
            (m[i] | baseline_mask[i]).all()
            for i in range(bright_mask.shape[0])
        ]
    )
    return result

@njit
def unique_min_per_filter(
    n_bright,
    n_baseline,
    n_filters_req,
    min_per_filter
):
    bright_mask = n_bright > (min_per_filter - 1)
    baseline_mask = n_baseline > 1

    enough_filters = evaluate_enough_filters(
        bright_mask,
        min_per_filter,
        n_filters_req
    )
    bright_and_baseline = evaluate_bright_and_baseline(
        bright_mask,
        baseline_mask
    )
    result = enough_filters & bright_and_baseline
    return result

@njit
def unique_min_samples(
    n_bright,
    n_baseline,
    n_filters_req,
    n_samples_req
):
    enough_samples = n_bright.sum() > n_samples_req
    result = (
        enough_samples & unique_min_per_filter(
            n_bright,
            n_baseline,
            n_filters_req,
            1
        )
    )
    return result

def unique_min_per_filter_factory(n_filters_req, min_per_filter):
    @njit
    def achromatic_func(n_bright, n_baseline):
        result = unique_min_per_filter(
            n_bright,
            n_baseline,
            n_filters_req,
            min_per_filter
        )

        return result

    return achromatic_func  

def unique_min_samples_factory(n_filters_req, n_samples_req):
    @njit
    def achromatic_func(n_bright, n_baseline):
        result = unique_min_samples(
            n_bright,
            n_baseline,
            n_filters_req,
            n_samples_req
        )
        return result

    return achromatic_func
