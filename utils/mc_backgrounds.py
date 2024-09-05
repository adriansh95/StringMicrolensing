"""This module provides the functions used in monte_carlo_lensing_nsc.ipynb"""
import numpy as np
import logging

def synthesize_background(lightcurves, rates, taus, rng):
    """Assumes df is sorted by its time column (more docstring to come)"""
    l = ["objectid", "filter", "mag_auto", "magerr_auto", "mjd"]
    m = np.isfinite(rates)
    durations = taus[m]
    ev_rates = rates[m]
    g = lightcurves[l].groupby(by="objectid", sort=False, group_keys=False)
    synth_mags = g.apply(_synthesize_background, ev_rates, durations, rng)
    result = lightcurves.assign(mag_auto=synth_mags)
    return result

def _synthesize_background(df, rates, taus, rng):
    logging.info(f"objectid = {df.iloc[0, 0]}, rng_state = {rng.bit_generator.state['state']['state']}")
    t_last_sample = df.iloc[-1, 4]
    t_first_sample = df.iloc[0, 4]
    event_time = (t_last_sample - t_first_sample) + taus
    n_ev_exp = event_time * rates
    n_ev_drawn = rng.poisson(lam=n_ev_exp)
    result = df.iloc[:, 2].copy()

    for i in range(len(taus)):
        n_ev = n_ev_drawn[i]
        tau = taus[i]
        start_times = rng.uniform(low=t_first_sample-tau, high=t_last_sample, size=n_ev)
    
        for st in start_times:
            lower_bound = st
            upper_bound = st + tau
            start_idx = df["mjd"].searchsorted(lower_bound, side="left")
            stop_idx = df["mjd"].searchsorted(upper_bound, side="right")
            result.iloc[start_idx:stop_idx] += (-2.5 * np.log10(2))

    return result