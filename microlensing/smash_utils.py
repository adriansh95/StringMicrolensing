import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datalab_utils import weighted_mean, weighted_std

def aggregator(df, column_list):
    try:
        true_value = df.loc[df["excursion"], column_list].iloc[0].squeeze()
    except IndexError:
        true_value = np.nan
    false_value = df.loc[~df["excursion"], column_list].mean(axis=0)
    result = true_value - false_value
    return result

def aggregator2(df):
    categories = np.array([(False, False), (False, True), (True, False), (True, True)])
    m = df["excursion"]
    if m.any():
        excursion_value = df.loc[m, "forced"].iloc[0]
        none_forced = ~(df.loc[~m, "forced"].any())
        all_forced = df.loc[~m, "forced"].all()
        if (none_forced | all_forced):
            if excursion_value:
                other_value = excursion_value & all_forced
            else:
                other_value = excursion_value & none_forced
            result = np.where((categories == (excursion_value, other_value)).all(axis=1))[0][0]
        else:
            result = 5
    else:
        result = np.nan
    return result

def delta_aggregator(df, data_column, err_column):
    m_ex = df["excursion"]
    samples = df.loc[~m_ex, data_column].values
    weights = (df.loc[~m_ex, err_column].values / 60**2)**-2
    mean = weighted_mean(samples, weights)
    delta = (df.loc[m_ex, data_column] - mean) * 60**2
    return np.abs(delta)


def plot_lightcurve(lightcurve_df, object_id, 
                    fig_dir="/dlusers/adriansh/work/analyses/plots/strings/", 
                    savefig=False):
    colors = np.array(["tab:purple", "tab:green", "tab:orange", "tab:red", "tab:brown"])
    filters = np.array(['u', 'g', 'r', 'i', 'z'])
    mask_filters = np.array([lightcurve_df["filter"] == f for f in filters])
    min_mjd = lightcurve_df["mjd"].astype(int).min()

    fig, ax = plt.subplots(figsize=(12, 9))

    object_id_str = str(object_id).replace('.', '-')
    mask_id = lightcurve_df["id"] == object_id
    source_lc_df = lightcurve_df.loc[mask_id]

    filters_with_excursion = source_lc_df.loc[lightcurve_df["excursion"], "filter"]
    
    for f in filters_with_excursion:
        m = filters == f
        c = colors[m][0]
        mask_filter = mask_filters[m]
        mask = mask_id & mask_filter
        mask_exc = mask & lightcurve_df["excursion"]
        
        lightcurve_full = lightcurve_df.loc[mask]

        lightcurve_exc = lightcurve_df.loc[mask_exc]
        
        vals = lightcurve_df.loc[(mask & ~lightcurve_df["excursion"]), "cmag"].values
        weights = (lightcurve_df.loc[(mask & ~lightcurve_df["excursion"]), "cerr"].values)**-2
        w_mean = weighted_mean(vals, weights)
        w_sigma = weighted_std(vals, weights)
        delta = lightcurve_exc["cmag"].values[0] - w_mean
        ratio = np.power(10, -0.4 * delta)
        n_sigma = delta / (lightcurve_exc["cerr"].values[0] + w_sigma)

        ax.errorbar(lightcurve_full["mjd"] - min_mjd, lightcurve_full['cmag'], 
                    yerr=lightcurve_full["cerr"],
                    marker='.', ms=8, capsize=5, color=c, ls="None", alpha=0.3)

        ax.errorbar(lightcurve_exc["mjd"] - min_mjd, lightcurve_exc['cmag'], yerr=lightcurve_exc["cerr"],
                    marker='.', ms=8, capsize=5, color=c, 
                    label=fr"{f}: {ratio:0.2f}, {n_sigma:0.2f}$\sigma$", ls="None")
        
        ax.axhline(w_mean, color=c, linestyle='--')
        ax.axhspan(w_mean - w_sigma, w_mean + w_sigma, color=c, alpha=0.3)

    ax.set_xlabel(f'MJD - {min_mjd}', fontsize=20)
    ax.set_ylabel('Calibrated Magnitude', fontsize=20)
    ax.legend(fontsize=18)
    ax.tick_params(labelsize=18)
    ax.set_title(f"Object {object_id:.6f}", fontsize=22)
    ax.invert_yaxis()
    plt.show(fig)
    
    if savefig:
        fig.savefig(f"{fig_dir}{object_id_str}_kde_detection_lightcurve.png")
    plt.close(fig)