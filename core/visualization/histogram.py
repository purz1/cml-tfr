import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
from typing import Optional, Tuple, Union, Dict
from pathlib import Path


def plot_histogram(
    df: pd.DataFrame,
    dependent_variable: str,
    x_variable: str,
    palette: Optional[Union[str, Dict[str, str]]] = None,
    bins: int = 10,
    kde: bool = True,
    line_width: float = 3,
    fontsize: int = 7,
    xlabel: str = "Eccentricity",
    ylabel: str = "Cell count",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (4, 3),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plots a histogram to compare the distribution of a continuous variable (`x_variable`)
    across categories defined by a `dependent_variable`.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data.
        dependent_variable (str): Column name representing categorical grouping (e.g., patient status).
        x_variable (str): Column name of the numeric variable to plot (e.g., eccentricity).
        palette (str or dict, optional): Color palette for different groups. Defaults to seaborn's default.
        bins (int): Number of bins for the histogram. Default is 10.
        kde (bool): Whether to overlay kernel density estimate lines. Default is True.
        line_width (float): Line width of KDE curves. Default is 3.
        fontsize (int): Base font size for labels and title. Default is 7.
        xlabel (str): Label for the x-axis. Default is "Eccentricity".
        ylabel (str): Label for the y-axis. Default is "Cell count".
        title (str, optional): Title of the plot. If None, no title is displayed.
        figsize (tuple): Figure size in inches. Default is (4, 3).
        save_path (str or Path, optional): If provided, saves the plot.

    Returns:
        None. Displays a matplotlib plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.histplot(
        data=df,
        x=x_variable,
        hue=dependent_variable,
        palette=palette,
        bins=bins,
        kde=kde,
        line_kws={"linewidth": line_width},
        edgecolor='none',
        ax=ax,
    )

    legend = ax.get_legend()
    if legend:
        legend.set_title(None)
        for text in legend.get_texts():
            text.set_fontsize(fontsize - 1)
        legend.get_frame().set_edgecolor('none')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='x', labelsize=fontsize - 1)
    ax.tick_params(axis='y', labelsize=fontsize - 1)

    # Set axis labels
    ax.set_xlabel(xlabel, fontsize=fontsize - 1)
    ax.set_ylabel(ylabel, fontsize=fontsize - 1)

    # Set title if provided
    if title:
        ax.set_title(title, fontsize=fontsize)

    # Show plot
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.show()
