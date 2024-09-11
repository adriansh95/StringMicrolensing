import matplotlib.pyplot as plt
import numpy as np

def plot_lightcurve(lightcurve_df, ax, **kwargs):
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:brown", 'k']
    filters = ['u', 'g', 'r', 'i', 'z', 'Y']
    x_column = kwargs.get("time_column", "mjd")
    xerr_column = kwargs.get("exptime_column", "exptime")
    yerr_column = kwargs.get("magerr_column", "magerr_auto")
    y_column = kwargs.get("mag_column", "mag_auto")
    filter_column = kwargs.get("filter_column", "filter")
    xerr = np.vstack((np.zeros(len(lightcurve_df)), lightcurve_df[xerr_column].values))

    for f, c in zip(filters, colors):
        m_f = lightcurve_df[filter_column] == f
        lc = lightcurve_df.loc[m_f]
        xerr = np.vstack((np.zeros(len(lc)), lc[xerr_column] / 86400))

        ax.errorbar(lc[x_column], lc[y_column], xerr=xerr,
                    yerr=lc[yerr_column], marker='.', ms=8,
                    capsize=5, color=c, ls="None", label=f)

    ax.set_ylabel('Mag', fontsize=20)
    ax.tick_params(labelsize=18)
    ax.invert_yaxis()
    ax.grid(visible=True)

def plot_event(event_df, lightcurve_df, fig=None, **kwargs):
    """Plot full view of the lightcurve and a zoomed-in region to see the bright sequences"""
    id_column = kwargs.get("id_column", "objectid")
    objectid = lightcurve_df[id_column].iloc[0]

    if fig is None:
        fig, axes = plt.subplots(1, 1, figsize=(18, 9))
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
    first_event = event_df.loc[0, "t_start_min"]
    t_delta = last_event - first_event
    axes[0].legend(fontsize=18, loc="upper right")
    axes[1].set_xlabel(f'MJD', fontsize=20)
    tlims = ((first_event - 0.25 * t_delta), (last_event + 0.25 * t_delta))
    axes[1].set_xlim(tlims)
    fig.suptitle(f"Object {objectid}", fontsize=22)
    return fig
