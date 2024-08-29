@njit
def time_lensable(mjds, exp_times, filters, taus):
    """exp_times assumed to be in same units as mjds (days),
    filters are cast as type U1"""
    n_samples = len(mjds)
    all_filter_counts = np.zeros(7)
    result = np.zeros(taus.shape)

    for f in filters:
        all_filter_counts[FILTER_ORDER[f]] += 1

    for i in range(1, n_samples-2):
        bright_filter_counts = np.zeros(7)
        bright_filter_counts[FILTER_ORDER[filters[i]]] += 1

        for j in range(i+1, n_samples-1):
            bright_filter_counts[FILTER_ORDER[filters[j]]] += 1
            baseline_filter_counts = all_filter_counts - bright_filter_counts
            baseline_filter_bool = baseline_filter_counts > 0
            bright_filter_bool = bright_filter_counts > 0
            enough_filters = bright_filter_bool.sum() > 1
            bright_and_baseline = bright_filter_bool[bright_filter_bool] == baseline_filter_bool[bright_filter_bool]

            if enough_filters & bright_and_baseline.all():
                ts = (mjds[i-1] + exp_times[i-1], mjds[i], mjds[j] + exp_times[j], mjds[j+1])
                tau_min = ts[2] - ts[1]
                tau_max = ts[3] - ts[0]
                tau_mask = (tau_min < taus) & (taus < tau_max)
                result[tau_mask] += t_of_tau(taus[tau_mask], ts)

    return result


@njit
def _find_t_next_other_filter(i, exp_ends, filters):
    this_filter = filters[i]
    result = np.inf

    for j in range(i + 1, len(exp_ends)):

        if filters[j] != this_filter:
            result = exp_ends[j]
            break

    return result

@njit
def _measure_time(t_this, t_floor, t_next_other_filter, taus):
    t = np.maximum(t_floor, t_next_other_filter)
    result = np.maximum(t_this + taus - t, 0)
    return result

@njit
def _compute_t_floor(t_this, taus, t_next_other_filter):
    result = np.maximum(t_this + taus, t_next_other_filter)
    return result

def measure_time(df, taus, filter_column="filter", mjd_column="mjd", exptime_column="exptime"):
    filters = df[filter_column].values.astype("U1")
    mjds = df[mjd_column].values
    exp_times = df[exptime_column].values / SECONDS_PER_DAY
    result = _measure_total_time(taus, filters, mjds, exp_times)
    return result

@njit
def _measure_total_time(taus, filters, mjds, exp_times):
    exp_ends = mjds + exp_times
    t_floor = np.zeros(taus.shape)
    result = np.zeros(taus.shape)

    for i in range(len(mjds) - 1):
        t_this = mjds[i]
        t_next_other_filter = _find_t_next_other_filter(i, exp_ends, filters)

        if np.isfinite(t_next_other_filter):
            result += _measure_time(t_this, t_floor, t_next_other_filter, taus)
        else:
            break

        t_floor = _compute_t_floor(t_this, taus, t_next_other_filter)

    result -= _subtract_time(taus, mjds, exp_ends)
    return result

@njit
def _subtract_time(taus, mjds, exp_ends):
    result = np.maximum(taus - (exp_ends[-1] - mjds[0]), 0)
    return result

def _check_time_contiguity(df):
    result = False
    cl = df["cluster_label"]
    n_bright = len(cl) - cl.sum()

    if n_bright > 0:
        mask_baseline = df["cluster_label"].values.astype(bool)    
        n_total = len(df)
        idxs = np.arange(n_total)
        baseline_idxs = idxs[mask_baseline]
        idx_diffs = np.diff(baseline_idxs)
        boundary_idxs = np.where(idx_diffs == (n_bright + 1))[0]
        case1 = len(boundary_idxs) == 1
        case2 = baseline_idxs[0] == n_bright
        case3 = (n_total - 1) - baseline_idxs[-1] == n_bright

        if any([case1, case2, case3]):
            result = True

    return result

def subtract_baseline(df, mag_column="mag_auto", magerr_column="magerr_auto"):
    df_grouped = df.groupby(by=["objectid", "filter"], group_keys=False, sort=False)
    result = df_grouped.apply(_subtract_baseline_apply, mag_column, magerr_column)
    return result

def _subtract_baseline_apply(df, mag_column, magerr_column):
    mask_baseline = df["cluster_label"].values.astype(bool)
    samples_baseline = df.loc[mask_baseline, mag_column].values
    weights_baseline = df.loc[mask_baseline, magerr_column].values**-2
    baseline = np.average(samples_baseline, weights=weights_baseline)
    baseline_err = np.sqrt(1 / weights_baseline.sum())
    result = df.assign(delta_mag=df[mag_column] - baseline,
                       delta_mag_err=df[magerr_column] + baseline_err)
    return result
