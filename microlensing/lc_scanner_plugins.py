import numpy as np
from numba import njit

@njit
def numba_achromatic(
    n_bright,
    n_baseline,
    n_filters_req,
    min_per_filter
):
    bright_mask = n_bright > (min_per_filter - 1)
    baseline_mask = n_baseline > 1
    enough_filters = (bright_mask.sum(axis=1) > (n_filters_req - 1))

    # In case I choose not to use Numba here,
    #     bright_and_baseline = (~bright_filter_mask | baseline_filter_mask).all(axis=1)
    # is equivalent but quite a bit slower.
    m = ~bright_mask
    bright_and_baseline = [
        (m[i] | baseline_mask[i]).all() for i in range(n_bright.shape[0])
    ]
    bright_and_baseline = np.array(bright_and_baseline)
    result = enough_filters & bright_and_baseline
    return result

def achromatic_func_factory(n_filters_req, min_per_filter):
    @njit
    def achromatic_func(
        n_bright,
        n_baseline,
    ):
        result = numba_achromatic(
            n_bright,
            n_baseline,
            n_filters_req,
            min_per_filter
        )

        return result

    return achromatic_func  
