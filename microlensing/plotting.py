import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

TIMESCALES = {
    "values_in_seconds": np.array(
        [60, 3600, 86400, 2592000, 31536000]
    ),
    "labels": ["1 min", "1 hr", "1 day", "1 mo", "1 yr"]
}

def plot_kde(df, ax, mag_column="mag_auto"):
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:brown", 'k']
    filters = ['u', 'g', 'r', 'i', 'z', 'Y']
    markers = ['.', '^', 'v', '+', 'x', 'D']
    color_key = dict(zip(filters, colors))
    marker_key = dict(zip(filters, markers))
    g = df.groupby(by="filter")

    for f, group in g:
        samples = group[mag_column]
 
        if len(samples) == 0:
            continue

        weights = group["magerr_auto"]**-2
        kde = gaussian_kde(samples, bw_method=1, weights=weights)
        bw = np.sqrt(np.mean(weights**(-1)))
        kde.set_bandwidth(bw / np.sqrt(kde.covariance[0, 0]))
        low = samples.min() - 0.5
        high = samples.max() + 0.5
        x = np.linspace(low, high, num=1001)
        y = kde(x)
        ax.plot(y, x, color=color_key[f])
        maxima = find_peaks(y)[0]
        ax.scatter(
            y[maxima],
            x[maxima],
            color=color_key[f],
            marker=marker_key[f]
        )

def plot_lightcurve(lightcurve_df, ax, **kwargs):
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:brown", 'k']
    filters = ['u', 'g', 'r', 'i', 'z', 'Y']
    markers = ['.', '^', 'v', '+', 'x', 'D']
    x_column = kwargs.get("time_column", "mjd")
    xerr_column = kwargs.get("exptime_column", "exptime")
    yerr_column = kwargs.get("magerr_column", "magerr_auto")
    y_column = kwargs.get("mag_column", "mag_auto")
    filter_column = kwargs.get("filter_column", "filter")
    xerr = np.vstack((np.zeros(len(lightcurve_df)), lightcurve_df[xerr_column].values))

    for f, c, m in zip(filters, colors, markers):
        m_f = lightcurve_df[filter_column] == f
        lc = lightcurve_df.loc[m_f]
        xerr = (lc[xerr_column] / 86400) / 2

        if len(lc) == 0:
            continue

        container = ax.errorbar(
            lc[x_column] + xerr,
            lc[y_column],
            xerr=xerr,
            yerr=lc[yerr_column],
            ms=8,
            capsize=5,
            color=c,
            marker=m,
            ls="None",
            label="_nolegend_"
        )
        container[0].set_label(f)

    ax.tick_params(labelsize=18)
    ax.invert_yaxis()
    ax.grid(visible=True)

def plot_event(event_df, lightcurve_df, fig=None, **kwargs):
    """Plot full view of the lightcurve and a zoomed-in region to see the bright sequences"""
    id_column = kwargs.get("id_column", "objectid")
    objectid = lightcurve_df[id_column].iloc[0]

    if fig is None:
        fig, axes = plt.subplots(2, 1, figsize=(18, 9))
    else:
        axes = fig.get_axes()

    for ax in axes:
        ax.cla()
        plot_lightcurve(lightcurve_df, ax, **kwargs)

        for _, event in event_df.iterrows():
            ev_start = event["t_start_min"]
            ev_end = event["t_end_min"]
            barlims = (ev_start, ev_end)
            ax.axvspan(barlims[0], barlims[1], alpha=0.2, color="tab:gray")

    last_event = event_df.loc[event_df.index[-1], "t_end_min"]
    first_event = event_df.loc[event_df.index[0], "t_start_min"]
    t_delta = last_event - first_event
    axes[0].legend(fontsize=18, loc="upper right")
    axes[1].set_xlabel(f'MJD', fontsize=20)
    tlims = ((first_event - 0.25 * t_delta), (last_event + 0.25 * t_delta))
    axes[1].set_xlim(tlims)
    fig.suptitle(f"Object {objectid}", fontsize=22)
    return fig
