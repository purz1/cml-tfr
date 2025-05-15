import numba
import numpy as np
import pandas as pd
import umap

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]

@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _red(x):
    return (x & 0xFF0000) >> 16

@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _green(x):
    return (x & 0x00FF00) >> 8

@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _blue(x):
    return x & 0x0000FF

_themes = {
    "fire": {
        "cmap": "fire",
        "color_key_cmap": "rainbow",
        "background": "black",
        "edge_cmap": "fire",
    },
    "viridis": {
        "cmap": "viridis",
        "color_key_cmap": "Spectral",
        "background": "black",
        "edge_cmap": "gray",
    },
    "inferno": {
        "cmap": "inferno",
        "color_key_cmap": "Spectral",
        "background": "black",
        "edge_cmap": "gray",
    },
    "blue": {
        "cmap": "Blues",
        "color_key_cmap": "tab20",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "red": {
        "cmap": "Reds",
        "color_key_cmap": "tab20b",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "green": {
        "cmap": "Greens",
        "color_key_cmap": "tab20c",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "darkblue": {
        "cmap": "darkblue",
        "color_key_cmap": "rainbow",
        "background": "black",
        "edge_cmap": "darkred",
    },
    "darkred": {
        "cmap": "darkred",
        "color_key_cmap": "rainbow",
        "background": "black",
        "edge_cmap": "darkblue",
    },
    "darkgreen": {
        "cmap": "darkgreen",
        "color_key_cmap": "rainbow",
        "background": "black",
        "edge_cmap": "darkpurple",
    },
}

_diagnostic_types = np.array(["pca", "ica", "vq", "local_dim", "neighborhood"])

def _get_embedding(umap_object):
    if hasattr(umap_object, "embedding_"):
        return umap_object.embedding_
    elif hasattr(umap_object, "embedding"):
        return umap_object.embedding
    else:
        raise ValueError("Could not find embedding attribute of umap_object")

def _get_metric(umap_object):
    if hasattr(umap_object, "metric"):
        return umap_object.metric
    else:
        # Assume euclidean if no attribute per cuML.UMAP
        return "euclidean"

def _select_font_color(background):
    if background == "black":
        font_color = "white"
    elif background.startswith("#"):
        mean_val = np.mean(
            [int("0x" + c) for c in (background[1:3], background[3:5], background[5:7])]
        )
        if mean_val > 126:
            font_color = "black"
        else:
            font_color = "white"

    else:
        font_color = "black"

    return font_color

def _matplotlib_points(
    points,
    ax=None,
    labels=None,
    values=None,
    cmap="Blues",
    color_key=None,
    color_key_cmap="Spectral",
    background="white",
    width=800,
    height=800,
    show_legend=True,
    alpha=None,
    point_size=None,
):
    """Use matplotlib to plot points"""
    if point_size is None:
        point_size = (100.0 / np.sqrt(points.shape[0]))

    legend_elements = None

    if ax is None:
        dpi = plt.rcParams["figure.dpi"]
        fig = plt.figure(figsize=(width / dpi, height / dpi))
        ax = fig.add_subplot(111)

    ax.set_facecolor(background)

    # Color by labels
    if labels is not None:
        if labels.shape[0] != points.shape[0]:
            raise ValueError(
                "Labels must have a label for "
                "each sample (size mismatch: {} {})".format(
                    labels.shape[0], points.shape[0]
                )
            )
        if color_key is None:
            unique_labels = np.unique(labels)
            num_labels = unique_labels.shape[0]
            color_key = plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
            legend_elements = [
                Patch(facecolor=color_key[i], label=unique_labels[i])
                for i, k in enumerate(unique_labels)
            ]

        if isinstance(color_key, dict):
            colors = pd.Series(labels).map(color_key)
            unique_labels = np.unique(labels)
            legend_elements = [
                Patch(facecolor=color_key[k], label=k) for k in unique_labels
            ]
        else:
            unique_labels = np.unique(labels)
            if len(color_key) < unique_labels.shape[0]:
                raise ValueError(
                    "Color key must have enough colors for the number of labels"
                )

            new_color_key = {
                k: matplotlib.colors.to_hex(color_key[i])
                for i, k in enumerate(unique_labels)
            }
            legend_elements = [
                Patch(facecolor=color_key[i], label=k)
                for i, k in enumerate(unique_labels)
            ]
            colors = pd.Series(labels).map(new_color_key)

        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=colors, alpha=alpha, linewidths=0)

    # Color by values
    elif values is not None:
        if values.shape[0] != points.shape[0]:
            raise ValueError(
                "Values must have a value for "
                "each sample (size mismatch: {} {})".format(
                    values.shape[0], points.shape[0]
                )
            )
        ax.scatter(
            points[:, 0], points[:, 1], s=point_size, c=values, cmap=cmap, alpha=alpha
        )

    # No color (just pick the midpoint of the cmap)
    else:

        color = plt.get_cmap(cmap)(0.5)
        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=color)


    return ax, legend_elements

def points(
    umap_object,
    points=None,
    labels=None,
    values=None,
    theme=None,
    cmap="Blues",
    color_key=None,
    color_key_cmap="Spectral",
    background="white",
    width=800,
    height=800,
    show_legend=True,
    legend_location="center left",
    show_parameters=False,
    subset_points=None,
    ax=None,
    alpha=None,
    fontsize=9,
    point_size=None,
    save_path=None,
):
    """Plot an embedding as points. Currently this only works
    for 2D embeddings. While there are many optional parameters
    to further control and tailor the plotting, you need only
    pass in the trained/fit umap model to get results. This plot
    utility will attempt to do the hard work of avoiding
    over-plotting issues, and make it easy to automatically
    colour points by a categorical labelling or numeric values.

    This method is intended to be used within a Jupyter
    notebook with ``%matplotlib inline``.

    Parameters
    ----------
    umap_object: trained UMAP object
        A trained UMAP object that has a 2D embedding.

    points: array, shape (n_samples, dim) (optional, default None)
        An array of points to be plotted. Usually this is None
        and so the original embedding points of the umap_object
        are used. However points can be passed explicitly instead
        which is useful for points manually transformed.

    labels: array, shape (n_samples,) (optional, default None)
        An array of labels (assumed integer or categorical),
        one for each data sample.
        This will be used for coloring the points in
        the plot according to their label. Note that
        this option is mutually exclusive to the ``values``
        option.

    values: array, shape (n_samples,) (optional, default None)
        An array of values (assumed float or continuous),
        one for each sample.
        This will be used for coloring the points in
        the plot according to a colorscale associated
        to the total range of values. Note that this
        option is mutually exclusive to the ``labels``
        option.

    theme: string (optional, default None)
        A color theme to use for plotting. A small set of
        predefined themes are provided which have relatively
        good aesthetics. Available themes are:
           * 'blue'
           * 'red'
           * 'green'
           * 'inferno'
           * 'fire'
           * 'viridis'
           * 'darkblue'
           * 'darkred'
           * 'darkgreen'

    cmap: string (optional, default 'Blues')
        The name of a matplotlib colormap to use for coloring
        or shading points. If no labels or values are passed
        this will be used for shading points according to
        density (largely only of relevance for very large
        datasets). If values are passed this will be used for
        shading according the value. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    color_key: dict or array, shape (n_categories) (optional, default None)
        A way to assign colors to categoricals. This can either be
        an explicit dict mapping labels to colors (as strings of form
        '#RRGGBB'), or an array like object providing one color for
        each distinct category being provided in ``labels``. Either
        way this mapping will be used to color points according to
        the label. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    color_key_cmap: string (optional, default 'Spectral')
        The name of a matplotlib colormap to use for categorical coloring.
        If an explicit ``color_key`` is not given a color mapping for
        categories can be generated from the label list and selecting
        a matching list of colors from the given colormap. Note
        that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    background: string (optional, default 'white)
        The color of the background. Usually this will be either
        'white' or 'black', but any color name will work. Ideally
        one wants to match this appropriately to the colors being
        used for points etc. This is one of the things that themes
        handle for you. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    width: int (optional, default 800)
        The desired width of the plot in pixels.

    height: int (optional, default 800)
        The desired height of the plot in pixels

    show_legend: bool (optional, default True)
        Whether to display a legend of the labels

    show_parameters: bool (optional, default True)
        Whether to display UMAP parameters.
        
    subset_points: array, shape (n_samples,) (optional, default None)
        A way to select a subset of points based on an array of boolean
        values.

    ax: matplotlib axis (optional, default None)
        The matplotlib axis to draw the plot to, or if None, which is
        the default, a new axis will be created and returned.

    alpha: float (optional, default: None)
        The alpha blending value, between 0 (transparent) and 1 (opaque).

    fontsize: int
        Fontsize of the legend labels.

    save_path: str (optional, default None)
        Path to save the figure.

    Returns
    -------
    result: matplotlib axis
        The result is a matplotlib axis with the relevant plot displayed.
        If you are using a notebooks and have ``%matplotlib inline`` set
        then this will simply display inline.
    """
    if theme is not None:
        cmap = _themes[theme]["cmap"]
        color_key_cmap = _themes[theme]["color_key_cmap"]
        background = _themes[theme]["background"]

    if labels is not None and values is not None:
        raise ValueError(
            "Conflicting options; only one of labels or values should be set"
        )

    if alpha is not None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0 and 1 inclusive")

    if points is None:
        points = _get_embedding(umap_object)

    if subset_points is not None:
        if len(subset_points) != points.shape[0]:
            raise ValueError(
                "Size of subset points ({}) does not match number of input points ({})".format(
                    len(subset_points), points.shape[0]
                )
            )
        points = points[subset_points]

        if labels is not None:
            labels = labels[subset_points]
        if values is not None:
            values = values[subset_points]

    if points.shape[1] != 2:
        raise ValueError("Plotting is currently only implemented for 2D embeddings")

    font_color = _select_font_color(background)

    if ax is None:
        dpi = plt.rcParams["figure.dpi"]
        fig = plt.figure(figsize=(width / dpi, height / dpi))
        ax = fig.add_subplot(111)

    ax, legend_elements = _matplotlib_points(
        points,
        ax,
        labels,
        values,
        cmap,
        color_key,
        color_key_cmap,
        background,
        width,
        height,
        show_legend,
        alpha,
        point_size,
    )

    ax.set(xticks=[], yticks=[])

    legend_order =  list(color_key.keys())

    legend_elements = sorted(
        legend_elements, 
        key=lambda handle: legend_order.index(handle.get_label())
    )

    # Adjust legend position
    if show_legend and legend_elements is not None:
        if legend_location == "best":
            ax.legend(
                handles=legend_elements,
                loc=legend_location,
                frameon=False,
                fontsize=fontsize,
            )
        else:
            ax.legend(
                handles=legend_elements,
                loc=legend_location,
                bbox_to_anchor=(1, 0.5),
                frameon=False,
                fontsize=fontsize,
            )

    if show_parameters:
        ax.text(
            0.99,
            0.01,
            "UMAP: metric={}, n_neighbors={}, min_dist={}".format(
                _get_metric(umap_object), umap_object.n_neighbors, umap_object.min_dist
            ),
            transform=ax.transAxes,
            horizontalalignment="right",
            color=font_color,
        )

    ax.set_axis_off()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        
    return ax, legend_elements
