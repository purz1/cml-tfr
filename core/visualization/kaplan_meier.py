import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
from typing import Optional, Dict, Tuple, Union
from pathlib import Path


def plot_kaplan_meier_curve(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str,
    fontsize: int = 7,
    custom_legends: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    xlabel: str = "Time",
    ylabel: str = "Survival Probability",
    xlim: Tuple[float, float] = (0, 48),
    ylim: Tuple[float, float] = (0, 1.13),
    figsize: Tuple[float, float] = (3, 2.5),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    df = df.copy()
    df["_group_str"] = df[group_col].astype(str)
    group_order = list(custom_legends.keys()) if custom_legends else sorted(df["_group_str"].unique())

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axhline(y=0.5, color="black", linestyle='--', linewidth=1, zorder=1)

    fitters = []
    for group in group_order:
        group_data = df[df["_group_str"] == group]
        if group_data.empty:
            continue

        kmf = KaplanMeierFitter()
        label = custom_legends.get(group, group) if custom_legends else group
        color = colors.get(group, None) if colors else None

        kmf.fit(group_data[time_col], event_observed=group_data[event_col], label=label)
        fitters.append(kmf)

        kmf.plot_survival_function(
            ci_show=False, ax=ax, at_risk_counts=False,
            show_censors=True, color=color, linewidth=2,
        )

        final_prob = kmf.survival_function_at_times(group_data[time_col].max()).item()
        print(f"Probability ({label}) = {round(final_prob, 3)}")

    stats = multivariate_logrank_test(df[time_col], df["_group_str"], df[event_col])
    ax.text(xlim[1] * 0.8, 0.1, fr"$\it{{P}}$ = {stats.p_value:.3f}", fontsize=fontsize - 1)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=fontsize - 1)
    ax.set_ylabel(ylabel, fontsize=fontsize - 1)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 12))
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.01, 0.25))

    # Add at-risk table and reduce its vertical spacing
    at_risk_table = add_at_risk_counts(*fitters, ax=ax,
                                       xticks=np.arange(xlim[0], xlim[1] + 1, 12),
                                       rows_to_show=['At risk'],
                                       fontsize=fontsize - 1)

    ax.xaxis.set_tick_params(labelsize=fontsize - 1)
    ax.yaxis.set_tick_params(labelsize=fontsize - 1)

    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    legend_labels = [custom_legends.get(g, g) if custom_legends else g for g in group_order]
    legend_handles = [label_to_handle[l] for l in legend_labels if l in label_to_handle]

    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='upper right',
        prop={'size': fontsize - 1},
        frameon=False
    )

    ax.set_axisbelow(True)
    ax.spines[['right', 'top']].set_visible(False)

    if title:
        ax.set_title(title, fontsize=fontsize)

    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

    plt.show()
