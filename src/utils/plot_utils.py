# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 8
TEXTWIDTH = 505.89
LINEWIDTH = 239.39


def set_tex_params():
    plt.rcParams.update(
        {
            "font.family": "serif",  # use serif/main font for text elements
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            "pgf.texsystem": "pdflatex",
            "font.size": SMALL_SIZE,  # controls default text sizes
            "axes.titlesize": SMALL_SIZE,  # fontsize of the axes title
            "axes.labelsize": MEDIUM_SIZE,  # fontsize of the x and y labels
            "xtick.labelsize": SMALL_SIZE,  # fontsize of the tick labels
            "ytick.labelsize": SMALL_SIZE,  # fontsize of the tick labels
            "legend.fontsize": SMALL_SIZE,  # legend fontsize
            "figure.titlesize": BIGGER_SIZE,  # fontsize of the figure title})
        }
    )


def get_size(
    n_rows=1,
    n_cols=1,
    width="linewidth",
    fraction=1,
):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "linewidth":
        width_pt = LINEWIDTH
    elif width == "textwidth":
        width_pt = TEXTWIDTH
    else:
        raise ValueError("width parameter not known")

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (n_rows / n_cols)

    return (fig_width_in, fig_height_in)
