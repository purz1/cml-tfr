import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ranksums
from typing import Dict, Tuple


def calculate_foldchange_pvalue(
    df: pd.DataFrame,
    cat_vars: Dict[str, str],
    cont_vars: Dict[str, str],
    max_pvalue: float = 0.05
) -> pd.DataFrame:
    """
    Calculate log10 fold change and Wilcoxon p-values for all combinations of categorical and continuous variables.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data.
        cat_vars (dict): Dictionary of categorical column names and their display labels.
        cont_vars (dict): Dictionary of continuous column names and their display labels.
        max_pvalue (float): Maximum p-value to display; higher p-values are set to NaN.

    Returns:
        pd.DataFrame: DataFrame with columns ['Categorical', 'Continuous', 'Log10_FC', 'p-value'].
    """
    results = []

    for cat_col, cat_label in cat_vars.items():
        for cont_col, cont_label in cont_vars.items():
            if cat_col in df.columns and cont_col in df.columns:
                valid_df = df[[cat_col, cont_col]].dropna()

                grouped = valid_df.groupby(cat_col)[cont_col].apply(list)

                if len(grouped) == 2:
                    group_a, group_b = grouped.iloc[0], grouped.iloc[1]
                    median_a = np.median(group_a)
                    median_b = np.median(group_b)

                    # Avoid division by zero
                    if median_a > 0:
                        log10_fc = np.log10(median_b / median_a)
                    else:
                        log10_fc = np.nan

                    stat, p_value = ranksums(group_a, group_b)
                    p_value = p_value if p_value <= max_pvalue else np.nan

                    results.append((cat_label, cont_label, log10_fc, p_value))

    return pd.DataFrame(results, columns=['Categorical', 'Continuous', 'Log10_FC', 'p-value']).replace([np.inf, -np.inf], np.nan)



def plot_balloon_plot(
    df: pd.DataFrame,
    cat_vars: Dict[str, str],
    cont_vars: Dict[str, str],
    max_pvalue: float = 0.05,
    plot_height: int = 3,
    plot_width: int = 3,
    fontsize: int = 7,
    title: str = None,
    sort_by_pvalue: bool = False,
    save_path: str = None,
) -> None:
    """
    Creates a balloon plot where color encodes log10 fold-change and size encodes -log10(p-value)
    for categorical vs continuous variable combinations.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        cat_vars (Dict[str, str]): Mapping of categorical columns to display names.
        cont_vars (Dict[str, str]): Mapping of continuous columns to display names.
        max_pvalue (float): Maximum p-value threshold to plot.
        plot_height (int): Height of the plot in inches.
        plot_width (int): Width of the plot in inches.
        fontsize (int): Font size used in plot.
        title (str): Optional title for the plot.
        sort_by_pvalue (bool): Whether to sort the plot entries by p-value.
        save_path (str): Path to save the figure.
    """
    results_df = calculate_foldchange_pvalue(df, cat_vars, cont_vars, max_pvalue)

    if sort_by_pvalue:
        results_df = results_df.sort_values(by="p-value", ascending=False)

    vmax = np.abs(results_df['Log10_FC']).max()
    results_df['log_p'] = -np.log10(results_df['p-value'])
    scale_factor = min(plot_height, plot_width) + 1
    results_df['size_scaled'] = (scale_factor * results_df['log_p']) ** 2

    is_vertical = len(results_df["Categorical"].unique()) > 1
    x_col, y_col = ('Continuous', 'Categorical') if is_vertical else ('Categorical', 'Continuous')

    if not is_vertical:
        plot_width = 1

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    
    # Plot orientation
    scatter = ax.scatter(
        results_df[x_col], results_df[y_col],
        c=results_df['Log10_FC'],
        s=results_df['size_scaled'],
        edgecolors="black",
        cmap='coolwarm',
        vmin=-vmax,
        vmax=vmax,
        alpha=1
    )

    # Axis labels and formatting
    plt.xticks(rotation=45 if is_vertical else 0, ha='right' if is_vertical else 'center', fontsize=fontsize - 1)
    plt.yticks(fontsize=fontsize - 1)

    # Padding for layout
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    padding = 0.6
    ax.set_ylim(ymin - padding, ymax + padding)
    ax.set_xlim(xmin - padding, xmax + padding)

    # Colorbar
    norm = plt.Normalize(-vmax, vmax)
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("")
    cbar.ax.set_title(r"Log$_{10}$ FC", pad=10, fontsize=fontsize - 1, loc='left')
    cbar.ax.tick_params(labelsize=fontsize-1)

    # P-value legend (ball sizes)
    legend_p_values = [0.001, 0.01, 0.05]
    legend_elements = [
        Line2D([0], [0],
               marker='o',
               color='none',
               label=f'{p:.3f}',
               markerfacecolor='dimgrey',
               markersize=-scale_factor * np.log10(p),
               markeredgecolor="none")
        for p in legend_p_values
    ]

    ax.legend(
        handles=legend_elements,
        title=r"$\it{{P}}$-value",
        fontsize=fontsize - 1,
        loc='upper left',
        bbox_to_anchor=(1, 1),
        frameon=False,
        title_fontsize=fontsize - 1,
    )

    # Adjust colorbar position
    x0, y0, w, h = ax.get_position().bounds
    legend_x = x0 + w
    cbar.ax.set_position([legend_x + 0.05, 1 - y0 - h, 0.1, plot_height / 6])

    # Axis & title
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_axisbelow(True)
    ax.grid(axis='both', alpha=0.5, zorder=0)
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    if title:
        ax.set_title(title, fontsize=fontsize)

    #plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

    plt.show()