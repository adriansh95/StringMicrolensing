import matplotlib.pyplot as plt

def plot_event(event_df, lightcurve_df, objectid, fig=None, **kwargs):
    time_column = kwargs.get("time_column", "mjd")
    magerr_column = kwargs.get("magerr_column", "magerr_auto")
    mag_column = kwargs.get("mag_column", "mag_auto")
    id_column = kwargs.get("id_column", "objectid")
    filter_column = kwargs.get("filter_column", "filter")
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:brown", 'k']
    filters = ['u', 'g', 'r', 'i', 'z', 'Y']

    if fig is None:
        fig, axes = plt.subplots(2, 1, figsize=(18, 9))
    else:
        axes = fig.get_axes()

    for ax in axes:
        ax.cla()

    m_id = lightcurve_df[id_column] == objectid
    min_mjd = int(lightcurve_df.loc[m_id, time_column].min())

    for f, c in zip(filters, colors):
        m_f = lightcurve_df[filter_column] == f
        lc_full = lightcurve_df.loc[m_id & m_f]

        axes[0].errorbar(lc_full[time_column] - min_mjd, lc_full[mag_column], 
                         yerr=lc_full[magerr_column], marker='.', ms=8,
                         capsize=5, color=c, ls="None", label=f)
        axes[1].errorbar(lc_full[time_column] - min_mjd, lc_full[mag_column], 
                         yerr=lc_full[magerr_column], marker='.', ms=8,
                         capsize=5, color=c, ls="None")

    events = event_df.loc[(objectid, slice(None))]

    for ax in axes:
        ax.set_ylabel('Mag', fontsize=20)
        ax.tick_params(labelsize=18)
        ax.invert_yaxis()
        ax.grid(visible=True)

        for _, event in events.iterrows():
            ev_start = event["t_start_min"]
            ev_end = event["t_end_min"]
            barlims = (ev_start - min_mjd, ev_end - min_mjd)
            ax.axvspan(barlims[0], barlims[1], alpha=0.2, color="tab:gray")

    last_event = events.loc[events.index[-1], "t_end_min"]
    first_event = events.loc[0, "t_start_min"]
    t_delta = last_event - first_event
    axes[0].legend(fontsize=18, loc="upper right")
    axes[1].set_xlabel(f'MJD - {min_mjd}', fontsize=20)
    tlims = ((first_event - 0.25 * t_delta) - min_mjd, (last_event + 0.25 * t_delta) - min_mjd)
    axes[1].set_xlim(tlims)
    fig.suptitle(f"Object {objectid}", fontsize=22)
    return fig
